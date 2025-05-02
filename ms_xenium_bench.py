from pathlib import Path
import time
import pickle

from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import partipy as pt
import plotnine as pn
import matplotlib
import matplotlib.pyplot as plt

from data_utils import load_ms_xenium_data
from const import FIGURE_PATH, OUTPUT_PATH

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_bench_xenium"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_bench_xenium"
output_dir.mkdir(exist_ok=True, parents=True)

## setting up the optimization seetings
init_alg_list = pt.const.INIT_ALGS
optim_alg_list = [alg for alg in pt.const.OPTIM_ALGS if alg != "regularized_nnls"]
optim_settings_list = []
for init_alg in init_alg_list:
    for optim_alg in optim_alg_list:
        optim_settings_list.append(
            {
                "init_alg": init_alg,
                "optim_alg": optim_alg,
            }
        )
print(f"{len(optim_settings_list)=}")

## setting up different seeds to test
seed_list = [383329927, 3324115916, 2811363264, 1884968544, 1859786275, 
             3687649985, 369133708, 2995172877, 865305066, 404488628,
             2261209995, 4190266092, 3160032368, 3269070126, 3081541439, 
             3376120482, 2204291346, 550243861, 3606691181, 1934392872]

script_start_time = time.time()
print(f"### Start Time: {script_start_time}")

## downloading the data (or using cached data)
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
    ## NOTE: For Xenium data we do not need to select highly variable genes before PCA
    adata = atlas_adata[atlas_adata.obs[celltype_column]==celltype, :].copy()
    print("\n#####\n->", celltype, "\n", adata)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.pca(adata)

    ## some scanpy QC plots
    for qc_var in qc_columns:
        adata.obs[qc_var] = pd.Categorical(adata.obs[qc_var])
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=False, show=False, save=False)
    plt.savefig(figure_dir_celltype / "pca_var_explained.png")
    sc.pl.pca(adata, color=qc_columns, dimensions=[(0, 1), (0, 1)],
              ncols=2, size=8, alpha=0.50, show=False, save=False)
    plt.savefig(figure_dir_celltype / "pca_2D.png")

    ## for simplicity we will always use 10 principal components
    pt.set_dimension_aa(adata=adata, n_pcs=number_of_pcs_dict[celltype])

    ## check for the number of archetypes
    pt.var_explained_aa(adata=adata, min_a=archetypes_to_test[0], max_a=archetypes_to_test[-1], n_jobs=20)

    p = pt.plot_var_explained(adata)
    p.save(figure_dir_celltype / f"aa_var_explained.png", dpi=300)

    p = pt.plot_IC(adata)
    p.save(figure_dir_celltype / f"aa_IC.png", dpi=300)

    #print("Running the boostrap...")
    #pt.bootstrap_aa_multiple_k(adata=adata, n_archetypes_list=archetypes_to_test, n_bootstrap=10)
    #p = pt.plot_bootstrap_multiple_k(adata)
    #p.save(figure_dir_celltype / f"plot_bootstrap_multiple_k.png", dpi=300)

    ## QC plot for the number of archetypes in 2D
    pt.bootstrap_aa(adata=adata, n_bootstrap=20, n_archetypes=number_of_archetypes_dict[celltype], n_jobs=20)
    p = pt.plot_bootstrap_2D(adata)
    p.save(figure_dir_celltype / f"aa_bootstrap_2D.png", dpi=300)

    ## benchmark
    print("Running the benchmark...")
    for optim_dict in optim_settings_list:
        print("\n\t", optim_dict)
        optim_key = "__".join(list(optim_dict.values()))
        rss_trace_dict[celltype][optim_key] = {}
        pbar = tqdm(seed_list)
        for seed in pbar:
            pbar.set_description(f"Seed: {seed}")

            adata_bench = adata.copy()

            start_time = time.time()
            
            pt.compute_archetypes(adata_bench, 
                                  n_archetypes=number_of_archetypes_dict[celltype],
                                  init=optim_dict["init_alg"],
                                  optim=optim_dict["optim_alg"],
                                  weight=None,
                                  seed=seed,
                                  save_to_anndata=True,
                                  archetypes_only=False,
                                  verbose=False)
            
            end_time = time.time()
            execution_time = end_time - start_time

            rss_trace_dict[celltype][optim_key][seed] = adata_bench.uns["archetypal_analysis"]["RSS"]

            result_dict = {
                "celltype": celltype,
                "time": execution_time,
                "rss": adata_bench.uns["archetypal_analysis"]["RSS"][-1],
                "varexpl": adata_bench.uns["archetypal_analysis"]["varexpl"],
                "seed": seed,
                "n_samples": adata_bench.shape[0],
                "n_dimensions": number_of_pcs_dict[celltype],
                "n_archetypes": number_of_archetypes_dict[celltype],
            }
            result_dict = result_dict | optim_dict
            result_list.append(result_dict)

result_df = pd.DataFrame(result_list)
result_df.to_csv(output_dir / "results.csv", index=False)

with open(output_dir / "rss_trace_dict.pkl", "wb") as f:
    pickle.dump(rss_trace_dict, f)

## plot for the optimization results
# result_df = pd.read_csv(Path(OUTPUT_PATH) / "ms_bench" / "results.csv")
result_df["key"] = [init + "__" + optim for init, optim in zip(result_df["init_alg"], result_df["optim_alg"])]
settings = ["celltype", "init_alg", "optim_alg"]
features = ["time", "rss", "varexpl"]
agg_df = result_df.groupby(settings).agg({f: ["mean", "std"] for f in features})
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df = agg_df.reset_index()

p = (
    pn.ggplot(agg_df)
    + pn.geom_point(pn.aes(x="time_mean", y="rss_mean", color="optim_alg", shape="init_alg"), size=4)
    + pn.geom_errorbar(
        pn.aes(x="time_mean", ymin="rss_mean - rss_std", ymax="rss_mean + rss_std", color="optim_alg"),
        width=0.1)
    + pn.geom_errorbarh(
        pn.aes(y="rss_mean", xmin="time_mean - time_std", xmax="time_mean + time_std", color="optim_alg"),
        height=0.1)
    + pn.facet_wrap(facets="celltype", ncol=3, scales="free")
    + pn.labs(x="Time (s)", y="Residual Sum of Squares (RSS)", 
              color="Optimization Algorithm", shape="Initialization Algorithm") 
    + pn.theme_bw() 
    + pn.theme(figure_size=(12, 6)) 
    + pn.scale_color_manual(values={"projected_gradients": "green", "frank_wolfe": "blue"})
)
p.save(figure_dir / f"result.png", dpi=300)

## plot for the optimization traces
#with open(Path(OUTPUT_PATH) / "ms_bench" / "rss_trace_dict.pkl", "rb") as f:
#    rss_trace_dict = pickle.load(f)

celltype_list = list(rss_trace_dict.keys())

for celltype in celltype_list:

    celltype_result_list = []

    rss_trace_dict_celltype = rss_trace_dict[celltype]

    result_df_subset = result_df.loc[result_df["celltype"]==celltype, :]
    n_samples = result_df_subset["n_samples"].to_numpy()[0]
    n_dimensions = result_df_subset["n_dimensions"].to_numpy()[0]
    n_archetypes = result_df_subset["n_archetypes"].to_numpy()[0]
    figure_title = f"{celltype} | n_cells = {n_samples} | n_dimensions = {n_dimensions} | n_archetypes = {n_archetypes}"
    
    for optim_str, rss_per_seed_dict in rss_trace_dict_celltype.items():


        for seed, trace in rss_per_seed_dict.items():
            tmp_df = pd.DataFrame({"optim": optim_str, "seed": seed, "iter": np.arange(len(trace)), "rss_trace": trace})
            celltype_result_list.append(tmp_df)

    df = pd.concat(celltype_result_list)
    df["seed"] = pd.Categorical(df["seed"])

    df["optim"] = pd.Categorical(df["optim"], 
                                categories=[b + "__" + a for a in pt.const.OPTIM_ALGS for b in pt.const.INIT_ALGS])
    
    p = (pn.ggplot(df)
        + pn.geom_hline(yintercept=min(df["rss_trace"]), color="black", alpha=0.5, size=1)
        + pn.geom_line(pn.aes(x="iter", y="rss_trace", color="seed"), alpha=1.0, size=0.5)
        + pn.scale_y_log10()
        + pn.facet_wrap("optim")
        + pn.theme_bw()
        + pn.theme(figure_size=(10, 6))
        + pn.labs(x="Iteration", y="Residual Sum of Squares (RSS)", color="seed", title=figure_title)
        )
    p.save(figure_dir / f"rss_trace_{celltype}.png", dpi=300)

script_end_time = time.time()
print(f"### End Time: {script_end_time}")
elapsed_time = np.round((script_end_time - script_start_time) / 60, 2)
print(f"### Elapsed Time: {elapsed_time} minutes")
