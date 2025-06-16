import os
import tarfile
import zipfile
import hashlib
from pathlib import Path

import requests
import scanpy as sc
import numpy as np
import pandas as pd

from const import DATA_PATH, EXPECTED_CHECKSUMS


def compute_partial_sha256(file_path: Path, chunk_size=20 * 1024 * 1024) -> str:
    """Compute a partial SHA256 hash from the start and end of the file."""
    sha256 = hashlib.sha256()
    file_size = file_path.stat().st_size

    with open(file_path, "rb") as f:
        # Read start
        sha256.update(f.read(chunk_size))
        
        # Read end
        if file_size > chunk_size:
            f.seek(-chunk_size, os.SEEK_END)
            sha256.update(f.read(chunk_size))

    return sha256.hexdigest()


def file_needs_download(file_path: Path, expected_hash: str) -> bool:
    if not file_path.exists():
        return True
    actual_hash = compute_partial_sha256(file_path)
    if actual_hash != expected_hash:
        print(f"Checksum mismatch for {file_path.name}: expected {expected_hash}, got {actual_hash}")
        return True
    return False


def load_ms_data(use_cache: bool = True, data_dir=Path(".") / DATA_PATH):
    data_dir.mkdir(exist_ok=True)

    # File URL to download
    url = "https://cells-test.gi.ucsc.edu/ms-subcortical-lesions/snrna-atlas/sn_atlas.h5ad"
    filename = data_dir / os.path.basename(url)

    # Download file if it does not already exist
    if file_needs_download(filename, EXPECTED_CHECKSUMS["sn_atlas.h5ad"]) or not use_cache:
        print(f"Downloading sn_atlas.h5ad to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded: {filename}")
    else:
        print(f"File already exists, skipping: {filename}")

    # Second file: GSE279183 tar file
    geo_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE279183&format=file"
    output_tar = data_dir / "GSE279183_RAW.tar"
    output_dir = data_dir / "GSE279183_extracted"

    # Download second file only if it does not already exist
    if file_needs_download(output_tar, EXPECTED_CHECKSUMS["GSE279183_RAW.tar"]) or not use_cache:
        print(f"Downloading GSE279183_RAW.tar to {output_tar}...")
        response = requests.get(geo_url, stream=True)
        if response.status_code == 200:
            with open(output_tar, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        else:
            print("Failed to download file:", response.status_code)
            exit(1)
    else:
        print(f"File already exists, skipping: {output_tar}")

    # Only extract if output directory does not exist
    if not output_dir.exists():
        print("Extracting all files from the archive...")
        output_dir.mkdir()
        with tarfile.open(output_tar, "r") as tar:
            tar.extractall(path=output_dir)
            print(f"Extracted {len(tar.getnames())} files.")
        print("Extraction complete. Files are saved in:", output_dir)
    else:
        print(f"Extraction directory already exists, skipping: {output_dir}")

    # get the gene info
    sn_atlas_var_info = pd.read_csv(output_dir/ "GSM8563681_CO37_features.tsv.gz", sep="\t", names=["gene_ids", "gene_name", "gene_type"])

    one_to_many_genes = (sn_atlas_var_info
                        .value_counts("gene_name")
                        .reset_index()
                        .query("count > 1")["gene_name"]
                        .to_list())

    unique_gene_ids_remove = (sn_atlas_var_info.
                            loc[sn_atlas_var_info["gene_name"].isin(one_to_many_genes), :]
                            .groupby("gene_name")
                            .last()
                            .reset_index())

    sn_atlas_var_info = sn_atlas_var_info.loc[~sn_atlas_var_info["gene_ids"].isin(unique_gene_ids_remove["gene_ids"].to_list()), :]

    # finally read the atlas
    sn_atlas = sc.read(data_dir / "sn_atlas.h5ad")
    sn_atlas.var["gene_name"] = sn_atlas.var.index.to_list().copy()
    sn_atlas.var = sn_atlas.var.join(sn_atlas_var_info.set_index("gene_name"), how="left")
    sn_atlas.obs = sn_atlas.obs.reset_index(names="cell_id").set_index("cell_id")
    sn_atlas.var = sn_atlas.var.reset_index().set_index("gene_ids")
    sn_atlas.X = sn_atlas.X.astype(np.int32)
    return sn_atlas


def load_hepatocyte_data(use_cache: bool = True):
    data_dir = Path(".") / DATA_PATH
    data_dir.mkdir(exist_ok=True)

    file_urls = {
        "design": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84498/suppl/GSE84498%5Fexperimental%5Fdesign.txt.gz",
        "counts": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE84nnn/GSE84498/suppl/GSE84498%5Fumitab.txt.gz"
    }

    file_paths = {}
    for key, url in file_urls.items():
        filename = data_dir / os.path.basename(url)
        file_paths[key] = filename

        if use_cache and filename.exists():
            print(f"File already exists, skipping: {filename}")
        else:
            print(f"Downloading {url} to {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")

    # Read metadata and count matrix
    obs = pd.read_csv(file_paths["design"], sep="\t").set_index("well")
    count_df = (pd.read_csv(file_paths["counts"], sep="\t")
                  .set_index("gene")
                  .T
                  .loc[obs.index, :])

    # Construct AnnData
    adata = sc.AnnData(
        X=count_df.values.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[c.split(";")[0] for c in count_df.columns])
    )

    # Filter lowly expressed genes
    adata = adata[:, adata.X.sum(axis=0) >= 20].copy()

    # Remove batches of likely non-hepatocytes
    adata = adata[~adata.obs["batch"].isin(["AB630", "AB631"])].copy()

    return adata


def load_hepatocyte_data_2(use_cache=True, data_dir=Path(".") / DATA_PATH):
    data_dir = Path(".") / DATA_PATH
    data_dir.mkdir(exist_ok=True)

    file_dicts = {
        "metadata": {
            "filename": "hepatocyte_meta.txt",
            "url": "https://zenodo.org/records/6035873/files/Single_cell_Meta_data.txt?download=1",
        },
        "counts": {
            "filename": "hepatocyte_counts.txt",
            "url": "https://zenodo.org/records/6035873/files/Single_cell_UMI_COUNT.txt?download=1",
        }
    }

    for file_dict in file_dicts.values():

        filepath = data_dir / file_dict["filename"]
        url = file_dict["url"]

        if use_cache and filepath.exists():
            print(f"File already exists, skipping: {filepath}")
        else:
            print(f"Downloading {url} to {filepath}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filepath}")
    count_tmp = pd.read_csv(data_dir / file_dicts["counts"]["filename"]).set_index("Gene_Name")
    meta_tmp = pd.read_csv(data_dir / file_dicts["metadata"]["filename"])
    meta_tmp = (meta_tmp.loc[meta_tmp["Cell_barcode"].isin(count_tmp.columns.to_list())]
                .set_index("Cell_barcode"))
    adata = sc.AnnData(X=count_tmp.values.copy().T.astype(np.float32),
                    var=pd.DataFrame(index=count_tmp.index.copy()),
                    obs=meta_tmp.loc[count_tmp.columns.to_numpy(), :].copy())
    del count_tmp, meta_tmp
    adata = adata[(adata.obs["time_point"] == 0) & (adata.obs["cell_type"] == "Hep"), :].copy()
    adata = adata[:, adata.X.sum(axis=0) > 0].copy()
    return adata


def load_ms_xenium_data(use_cache=True, data_dir=Path(".") / DATA_PATH):
    data_dir = Path(".") / DATA_PATH
    data_dir.mkdir(exist_ok=True)

    zip_filename = "MS_xenium_data_v5_with_images_tmap.h5ad.zip"
    h5ad_filename = "MS_xenium_data_v5_with_images_tmap.h5ad"
    
    zip_path = data_dir / zip_filename
    file_path = data_dir / h5ad_filename
    url = "https://zenodo.org/records/8037425/files/MS_xenium_data_v5_with_images_tmap.h5ad.zip?download=1"

    # Download if needed
    if file_needs_download(zip_path, EXPECTED_CHECKSUMS[zip_filename]) or not use_cache:
        print(f"Downloading from {url} to {zip_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Zip file already exists: {zip_path}")

    # Extract if H5AD missing or invalid
    if file_needs_download(file_path, EXPECTED_CHECKSUMS[h5ad_filename]) or not use_cache:
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)
        print(f"File extracted to {file_path}")
    else:
        print(f"Extracted H5AD file already valid: {file_path}")

    return sc.read_h5ad(file_path)


def load_lupus_data(use_cache=True, data_dir=Path(".") / DATA_PATH):
    import gc

    data_dir = Path(".") / DATA_PATH
    data_dir.mkdir(exist_ok=True)

    url = "https://datasets.cellxgene.cziscience.com/4532eea4-24b7-461a-93f5-fe437ee96f0a.h5ad"
    file_name = "4532eea4-24b7-461a-93f5-fe437ee96f0a.h5ad"
    file_path = data_dir / file_name

    # Download file if it does not already exist
    if file_needs_download(file_path, EXPECTED_CHECKSUMS[file_name]) or not use_cache:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        print(f"Downloaded: {file_path}")
    else:
        print(f"File already exists, skipping: {file_path}")

    adata = sc.read_h5ad(file_path)
    adata.obs["Status"] = adata.obs["disease_state"].map({
        "managed": "Managed",
        "na": "Healthy",
        "flare": "Flare",
        "treated": "Treated"
    })
    #adata = adata[adata.obs["author_cell_type"]=="ncM", :].copy() # only consider non-classical monocytes
    #adata = adata[adata.obs["Status"] != "Treated", :].copy() # remove samples with "treated" status
    # remove columns we don"t need
    adata.obs.drop(columns=["mapped_reference_annotation", "cell_type_ontology_term_id", "is_primary_data", 
                            "cell_state", "tissue_ontology_term_id", "development_stage_ontology_term_id", 
                            "tissue", "organism", "tissue_type", "suspension_type", "organism_ontology_term_id",
                            "assay_ontology_term_id", "suspension_enriched_cell_types", "suspension_uuid",
                            "self_reported_ethnicity_ontology_term_id", "disease_ontology_term_id",
                            "sex_ontology_term_id"], 
                            inplace=True)
    # create new index
    adata.obs.index = [s.split("-")[0] + "-" + str(len(s.split("-"))) + "-" + str(donor_id) 
                    for s, donor_id in zip(adata.obs.index, adata.obs["donor_id"].to_list())]
    # remove obsm we don't need
    del adata.obsm["X_pca"], adata.obsm["X_umap"], adata.uns
    gc.collect()

    # use the raw counts
    adata.X = adata.raw.X

    # use gene symbols instead of ensembl IDs
    assert len(adata.var["feature_name"]) == len(adata.var["feature_name"].unique())
    adata.var = adata.var.set_index("feature_name")

    # remove lowly expressed genes
    adata = adata[:, adata.X.sum(axis=0) >= 20].copy()

    # remove processing cohort 4.0
    adata = adata[adata.obs["Processing_Cohort"]!="4.0", :].copy() # remove processing cohort 4.0

    return adata
