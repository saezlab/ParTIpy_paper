from pathlib import Path
import time
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import partipy as pt
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance
import plotnine as pn
import matplotlib
import matplotlib.pyplot as plt

from data_utils import load_ms_xenium_data
from const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_xenium_coreset"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_xenium_coreset"
output_dir.mkdir(exist_ok=True, parents=True)

## setting up different seeds to test TODO: Change this
seed_list = SEED_DICT["m"]

coreset_fraction_list = 1 / (np.array([2**n for n in range(0, 8)]) * (25/16))
coreset_fraction_arr = np.zeros(len(coreset_fraction_list)+1)
coreset_fraction_arr[1:] = coreset_fraction_list
coreset_fraction_arr[0] = 1.0

script_start_time = time.time()
print(f"### Start Time: {script_start_time}")

atlas_adata = load_ms_xenium_data()
print(atlas_adata)

## qc settings
qc_columns = ["type_spec", "Level3"]

## remap the cell type annotation to broader categories
mapping_dict = {
    "MP/MiGl_1": "Myeloid",
    "MP/MiGl_2": "Myeloid",
    "vascular_MP_1":"Myeloid",
    "vascular_MP_2": "Myeloid",
    "vascular_MP_3": "Myeloid",
    "Vascular_1": "Vascular",
    "Vascular_2": "Vascular",
    "Astro_WM": "Astrocyte",
    "Astro_GM": "Astrocyte",
    "Astro_WM_DA": "Astrocyte",
    "Astro_GM_DA": "Astrocyte",
    "OLG_WM": "Oligo",
    "OLG_WM_DA": "Oligo",
    "OLG_GM": "Oligo",
    "OPC": "OPC",
    "OPC_DA": "OPC",
    "COP": "COP",
    "NFOL/MFOL": "NFOL",
    "Schw": "Schwann",
    "Endo": "Endothelial",
    "Neurons": "Neurons",
    "vascular_T-cell": "T_cell",
    "T-cell": "T_cell",
    "Ependymal": "Ependymal",
    "unknown": "unknown",
}

atlas_adata.obs["celltype"] = atlas_adata.obs["Level2"].map(mapping_dict)

celltype_column = "celltype"
celltype_labels = ["Oligo", "Astrocyte", "Myeloid", "Vascular", "Schwann", "OPC", "Endothelial", "T_cell"]
print(atlas_adata.obs.value_counts(celltype_column))

## number of archetypes per celltype
archetypes_to_test = list(range(2, 15))
number_of_archetypes_dict = {
    "Oligo": 4,
    "Astrocyte": 4,
    "Myeloid": 5,
    "Vascular": 5,
    "Schwann": 4,
    "OPC": 5,
    "Endothelial": 3,
    "T_cell": 4,
}
assert set(celltype_labels) == set(number_of_archetypes_dict.keys())
number_of_pcs_dict = {
    "Oligo": 10,
    "Astrocyte": 10,
    "Myeloid": 10,
    "Vascular": 10,
    "Schwann": 10,
    "OPC": 10,
    "Endothelial": 10,
    "T_cell": 10,
}
assert set(celltype_labels) == set(number_of_pcs_dict.keys())

## initialize list to save the benchmarking results
result_list = []
rss_trace_dict = {}

for celltype in celltype_labels:

    rss_trace_dict[celltype] = {}

    ## set up plotting directory per celltype
    figure_dir_celltype = figure_dir / celltype
    figure_dir_celltype.mkdir(exist_ok=True)

    ## subsetting and preprocessing per celltype
    adata = atlas_adata[atlas_adata.obs[celltype_column]==celltype, :].copy()
    print("\n#####\n->", celltype, "\n", adata)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, mask_var="highly_variable")

    ## some scanpy QC plots
    for qc_var in qc_columns:
        adata.obs[qc_var] = pd.Categorical(adata.obs[qc_var])
    sc.pl.highly_variable_genes(adata, log=False, show=False, save=False)
    plt.savefig(figure_dir_celltype / "highly_variable_genes.png")
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=False, show=False, save=False)
    plt.savefig(figure_dir_celltype / "pca_var_explained.png")
    sc.pl.pca(adata, color=qc_columns, dimensions=[(0, 1), (0, 1)],
              ncols=2, size=10, alpha=0.75, show=False, save=False)
    plt.savefig(figure_dir_celltype / "pca_2D.png")

    ## for simplicity we will always use 10 principal components
    pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimension=number_of_pcs_dict[celltype])

    # reference archetype
    adata_ref = adata.copy()
    pt.compute_archetypes(adata_ref, 
                          n_archetypes=number_of_archetypes_dict[celltype],
                          use_coreset=False,
                          seed=42,
                          save_to_anndata=True,
                          archetypes_only=False,
                          verbose=False)

    for coreset_fraction in coreset_fraction_arr:

        print(coreset_fraction)

        pbar = tqdm(seed_list)
        for seed in pbar:
            pbar.set_description(f"Seed: {seed}")

            adata_bench = adata.copy()

            start_time = time.time()
            
            pt.compute_archetypes(adata_bench, 
                                  n_archetypes=number_of_archetypes_dict[celltype],
                                  use_coreset=True if coreset_fraction < 1 else False,
                                  coreset_fraction=coreset_fraction,
                                  seed=seed,
                                  save_to_anndata=True,
                                  archetypes_only=False,
                                  verbose=False)
            
            end_time = time.time()
            execution_time = end_time - start_time


            Z = adata_ref.uns["AA_results"]["Z"].copy()
            Z_hat = adata_bench.uns["AA_results"]["Z"].copy()
            Z_hat = align_archetypes(Z, Z_hat)
            rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

            result_dict = {
                "celltype": celltype,
                "time": execution_time,
                "rss": adata_bench.uns["AA_results"]["RSS_full"],
                "varexpl": adata_bench.uns["AA_results"]["varexpl"],
                "mean_rel_l2_distance": np.mean(rel_dist_between_archetypes),
                "seed": seed,
                "coreset_fraction": coreset_fraction,
                "n_samples": adata_bench.shape[0],
                "n_dimensions": number_of_pcs_dict[celltype],
                "n_archetypes": number_of_archetypes_dict[celltype],
            }

            result_list.append(result_dict)

result_df = pd.DataFrame(result_list)
result_df.to_csv(output_dir / "results.csv", index=False)

## plot for the results
# result_df = pd.read_csv(Path(OUTPUT_PATH) / "ms_coreset" / "results.csv")

for celltype in celltype_labels:
    result_df_ct = result_df.loc[result_df["celltype"]==celltype, :].copy()

    p = (pn.ggplot(result_df_ct)
        + pn.geom_point(pn.aes(x="coreset_fraction", y="mean_rel_l2_distance"))
        + pn.geom_smooth(pn.aes(x="coreset_fraction", y="mean_rel_l2_distance"))
        + pn.scale_x_log10()
        )
    p.save(figure_dir / f"mean_rel_l2_distance_vs_coreset_fraction_{celltype}.png", dpi=300, verbose=False)
    p = (pn.ggplot(result_df_ct)
        + pn.geom_point(pn.aes(x="coreset_fraction", y="time"))
        + pn.geom_smooth(pn.aes(x="coreset_fraction", y="time"))
        + pn.scale_x_log10()
        + pn.scale_y_log10()
        )
    p.save(figure_dir / f"time_vs_coreset_fraction_{celltype}.png", dpi=300, verbose=False)
