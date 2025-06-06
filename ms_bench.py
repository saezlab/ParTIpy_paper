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

from data_utils import load_ms_data
from const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_bench"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_bench"
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
seed_list = SEED_DICT["l"]

script_start_time = time.time()
print(f"### Start Time: {script_start_time}")

## downloading the data (or using cached data)
atlas_adata = load_ms_data()
print(atlas_adata)

## qc settings
qc_columns = ["lesion_type", "subtype"]

## cell types to consider
celltype_column = "celltype"
celltype_labels = ["MG", "AS", "OL", "OPC", "NEU", "EC"]
print(atlas_adata.obs.value_counts(celltype_column))

## number of archetypes per celltype
archetypes_to_test = list(range(2, 15))
number_of_archetypes_dict = {
    "MG": 6,
    "AS": 11,
    "OL": 5,
    "OPC": 5,
    "NEU": 9,
    "EC": 4,
}
assert set(celltype_labels) == set(number_of_archetypes_dict.keys())
number_of_pcs_dict = {
    "MG": 10,
    "AS": 10,
    "OL": 10,
    "OPC": 10,
    "NEU": 10,
    "EC": 10,
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

    ## QC plots for number of PCs
    pt.compute_shuffled_pca(adata, mask_var="highly_variable")
    p = pt.plot_shuffled_pca(adata)
    p.save(figure_dir_celltype / f"shuffled_pca_plot.png", dpi=300)

    ## for simplicity we will always use 10 principal components
    pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=number_of_pcs_dict[celltype])

    ## check for the number of archetypes
    pt.compute_selection_metrics(adata=adata, min_k=archetypes_to_test[0], max_k=archetypes_to_test[-1], n_jobs=20)

    p = pt.plot_var_explained(adata)
    p.save(figure_dir_celltype / f"aa_var_explained.png", dpi=300)

    p = pt.plot_IC(adata)
    p.save(figure_dir_celltype / f"aa_IC.png", dpi=300)

    print("Running the boostrap...")
    pt.compute_bootstrap_variance(adata=adata, n_archetypes_list=archetypes_to_test, n_bootstrap=10)
    p = pt.plot_bootstrap_variance(adata)
    p.save(figure_dir_celltype / f"plot_bootstrap_multiple_k.png", dpi=300)

    ## QC plot for the number of archetypes in 2D
    p = pt.plot_bootstrap_2D(adata, n_archetypes=number_of_archetypes_dict[celltype])
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
                                  n_restarts=1,
                                  init=optim_dict["init_alg"],
                                  optim=optim_dict["optim_alg"],
                                  weight=None,
                                  seed=seed,
                                  save_to_anndata=True,
                                  archetypes_only=False, 
                                  verbose=False)
            
            end_time = time.time()
            execution_time = end_time - start_time

            rss_trace_dict[celltype][optim_key][seed] = adata_bench.uns["AA_results"]["RSS"]

            result_dict = {
                "celltype": celltype,
                "time": execution_time,
                "rss": adata_bench.uns["AA_results"]["RSS"][-1],
                "varexpl": adata_bench.uns["AA_results"]["varexpl"],
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
result_df["description"] = [celltype + " | " + str(n_samples) + " | " + str(n_arch) + " | " + str(n_dim) for celltype, n_samples, n_arch, n_dim in 
                            zip(result_df["celltype"], result_df["n_samples"], result_df["n_archetypes"], result_df["n_dimensions"])]
result_df["rss_norm"] = result_df["rss"] / (result_df["n_samples"] * result_df["n_dimensions"])
result_df["key"] = [init + "__" + optim for init, optim in zip(result_df["init_alg"], result_df["optim_alg"])]

# Grouping and aggregation
settings = ["description", "init_alg", "optim_alg"]
features = ["time", "rss", "varexpl", "rss_norm"]
agg_df = result_df.groupby(settings).agg({f: ["mean", "std"] for f in features})
agg_df.columns = ['__'.join(col).strip() for col in agg_df.columns.values]
agg_df = agg_df.reset_index()

# Precompute error bar limits
agg_df["rss__ymin"] = agg_df["rss__mean"] - agg_df["rss__std"]
agg_df["rss__ymax"] = agg_df["rss__mean"] + agg_df["rss__std"]
agg_df["rss_norm__ymin"] = agg_df["rss_norm__mean"] - agg_df["rss_norm__std"]
agg_df["rss_norm__ymax"] = agg_df["rss_norm__mean"] + agg_df["rss_norm__std"]
agg_df["time__xmin"] = agg_df["time__mean"] - agg_df["time__std"]
agg_df["time__xmax"] = agg_df["time__mean"] + agg_df["time__std"]


p = (
    pn.ggplot(agg_df)
    + pn.geom_point(pn.aes(x="time__mean", y="rss__mean", color="optim_alg", shape="init_alg"), size=4)
    + pn.geom_errorbar(
        pn.aes(x="time__mean", ymin="rss__ymin", ymax="rss__ymax", color="optim_alg"), width=0)
    + pn.geom_errorbarh(
        pn.aes(y="rss__mean", xmin="time__xmin", xmax="time__xmax", color="optim_alg"), height=0)
    + pn.facet_wrap(facets="description", ncol=3, scales="free")
    + pn.labs(x="Time (s)", y="Residual Sum of Squares (RSS)", 
              color="Optimization Algorithm", shape="Initialization Algorithm") 
    + pn.theme_bw() 
    + pn.theme(figure_size=(12, 6)) 
    + pn.scale_color_manual(values={"projected_gradients": "green", "frank_wolfe": "blue"})
)
p.save(figure_dir / f"rss_vs_time.png", dpi=300)

p = (
    pn.ggplot(agg_df)
    + pn.geom_point(pn.aes(x="time__mean", y="rss_norm__mean", color="optim_alg", shape="init_alg"), size=4)
    + pn.geom_errorbar(
        pn.aes(x="time__mean", ymin="rss_norm__ymin", ymax="rss_norm__ymax", color="optim_alg"), width=0)
    + pn.geom_errorbarh(
        pn.aes(y="rss_norm__mean", xmin="time__xmin", xmax="time__xmax", color="optim_alg"), height=0)
    + pn.facet_wrap(facets="description", ncol=3, scales="free")
    + pn.labs(x="Time (s)", y="Normalized Residual Sum of Squares (RSS)", 
              color="Optimization Algorithm", shape="Initialization Algorithm") 
    + pn.theme_bw() 
    + pn.theme(figure_size=(12, 6)) 
    + pn.scale_color_manual(values={"projected_gradients": "green", "frank_wolfe": "blue"})
)
p.save(figure_dir / f"normalized_rss_vs_time.png", dpi=300)

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
