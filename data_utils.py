import os
import requests
import tarfile
from pathlib import Path

import scanpy as sc
import numpy as np
import pandas as pd

from .const import DATA_PATH

def load_ms_data(use_cache: bool = True):
    data_dir = Path(".") / DATA_PATH
    data_dir.mkdir(exist_ok=True)

    # File URL to download
    url = "https://cells-test.gi.ucsc.edu/ms-subcortical-lesions/snrna-atlas/sn_atlas.h5ad"
    filename = data_dir / os.path.basename(url)

    # Download file if it does not already exist
    if not filename.exists() or not use_cache:
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
    if not output_tar.exists() or not use_cache:
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

    # Extract all files from the tar archive
    os.makedirs(output_dir, exist_ok=True)

    print("Extracting all files from the archive...")
    with tarfile.open(output_tar, "r") as tar:
        tar.extractall(path=output_dir)
        print(f"Extracted {len(tar.getnames())} files.")

    print("Extraction complete. Files are saved in:", output_dir)

    # get the gene info
    sn_atlas_var_info = pd.read_csv(output_dir/ "GSM8563681_CO37_features.tsv.gz", sep="\t", names=["gene_ids", "gene_name", "gene_type"])

    one_to_many_genes = (sn_atlas_var_info
                        .value_counts("gene_name")
                        .reset_index()
                        .query("count > 1")["gene_name"]
                        .to_list())
    one_to_many_genes

    unique_gene_ids_remove = (sn_atlas_var_info.
                            loc[sn_atlas_var_info["gene_name"].isin(one_to_many_genes), :]
                            .groupby("gene_name")
                            .last()
                            .reset_index())
    unique_gene_ids_remove

    sn_atlas_var_info = sn_atlas_var_info.loc[~sn_atlas_var_info["gene_ids"].isin(unique_gene_ids_remove["gene_ids"].to_list()), :]

    # finally read the atlas
    sn_atlas = sc.read(data_dir / "sn_atlas.h5ad")
    sn_atlas.var["gene_name"] = sn_atlas.var.index.to_list().copy()
    sn_atlas.var = sn_atlas.var.join(sn_atlas_var_info.set_index("gene_name"), how="left")
    sn_atlas.obs = sn_atlas.obs.reset_index(names="cell_id").set_index("cell_id")
    sn_atlas.var = sn_atlas.var.reset_index().set_index("gene_ids")
    sn_atlas.X = sn_atlas.X.astype(np.int32)
    return sn_atlas