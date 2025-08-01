from pathlib import Path
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import partipy as pt
from py_pcha import PCHA
import matplotlib
import plotnine as pn

from ..utils.data_utils import load_ms_xenium_data
from ..utils.const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "mx_xenium_pypcha"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "mx_xenium_pypcha"
output_dir.mkdir(exist_ok=True, parents=True)

## setting up different seeds to test
seed_list = SEED_DICT["s"]

script_start_time = time.time()
print(f"### Start Time: {script_start_time}")

atlas_adata = load_ms_xenium_data()
print(atlas_adata)

## qc settings
qc_columns = ["type_spec", "Level3"]

## remap the cell type annotation to broader categories
## remap the cell type annotation to broader categories
mapping_dict = {
    "MP/MiGl_1": "Myeloid",
    "MP/MiGl_2": "Myeloid",
    "vascular_MP_1": "Myeloid",
    "vascular_MP_2": "Myeloid",
    "vascular_MP_3": "Myeloid",
    # "Vascular_1": "Endothelial",
    # "Vascular_2": "Endothelial",
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
celltype_labels = [
    "Oligo",
    "Astrocyte",
    "Myeloid",
    "Schwann",
    "OPC",
    "Endothelial",
    "T_cell",
]
print(atlas_adata.obs.value_counts(celltype_column))

## number of archetypes per celltype
archetypes_to_test = list(range(2, 10))
number_of_archetypes_dict = {
    "Oligo": 4,
    "Astrocyte": 4,
    "Myeloid": 5,
    "Schwann": 4,
    "OPC": 5,
    "Endothelial": 3,
    "T_cell": 3,
}
assert set(celltype_labels) == set(number_of_archetypes_dict.keys())
number_of_pcs_dict = {
    "Oligo": 10,
    "Astrocyte": 10,
    "Myeloid": 10,
    "Schwann": 10,
    "OPC": 10,
    "Endothelial": 10,
    "T_cell": 10,
}
assert set(celltype_labels) == set(number_of_pcs_dict.keys())

## helper function
def time_and_evaluate(X, n_runs: int, n_archetypes: int, seed: int):
    """Time both methods and calculate RSS for given data X"""
    n_samples = X.shape[0]

    # Storage for results
    partipy_times = []
    pypcha_times = []
    partipy_rss = []
    pypcha_rss = []
    partipy_varexpl = []
    pypcha_varexpl = []

    TSS = np.sum(X * X)

    for _ in range(n_runs):
        
        # ParTIpy
        start = time.perf_counter()
        np.random.seed(seed=seed)
        AA_object = pt.AA(n_archetypes=n_archetypes, verbose=False, seed=seed)
        AA_object.fit(X)
        Z_partipy = AA_object.Z
        A_partipy = AA_object.A
        partipy_time = time.perf_counter() - start
        RSS_partipy = np.linalg.norm(np.dot(A_partipy, Z_partipy) - X) ** 2
        RSS_norm_partipy = RSS_partipy / n_samples
        varexpl_partipy = (TSS - RSS_partipy) / TSS

        partipy_times.append(partipy_time)
        partipy_rss.append(RSS_norm_partipy)
        partipy_varexpl.append(varexpl_partipy)

        # py_pcha
        start = time.perf_counter()
        np.random.seed(seed=seed)
        XC, S, C, SSE, varexpl = PCHA(X=X.T, noc=n_archetypes)
        Z_pypcha = np.asarray(XC.T)
        A_pypcha = np.asarray(S.T)
        pypcha_time = time.perf_counter() - start
        RSS_pypcha = np.linalg.norm(np.dot(A_pypcha, Z_pypcha) - X) ** 2
        RSS_norm_pypcha = RSS_pypcha / n_samples
        varexpl_pypcha = (TSS - RSS_pypcha) / TSS

        pypcha_times.append(pypcha_time)
        pypcha_rss.append(RSS_norm_pypcha)
        pypcha_varexpl.append(varexpl_pypcha)

    return {
        "partipy_time_mean": np.mean(partipy_times),
        "partipy_time_std": np.std(partipy_times),
        "pypcha_time_mean": np.mean(pypcha_times),
        "pypcha_time_std": np.std(pypcha_times),
        "partipy_rss_mean": np.mean(partipy_rss),
        "partipy_rss_std": np.std(partipy_rss),
        "pypcha_rss_mean": np.mean(pypcha_rss),
        "pypcha_rss_std": np.std(pypcha_rss),
        "partipy_varexpl_mean": np.mean(partipy_varexpl),
        "partipy_varexpl_std": np.std(partipy_varexpl),
        "pypcha_varexpl_mean": np.mean(pypcha_varexpl),
        "pypcha_varexpl_std": np.std(pypcha_varexpl),
    }

## initialize list to save the benchmarking results
result_list = []

N_RUNS = 3

for celltype in celltype_labels:
    ## set up plotting directory per celltype
    figure_dir_celltype = figure_dir / celltype
    figure_dir_celltype.mkdir(exist_ok=True)

    ## subsetting and preprocessing per celltype
    adata = atlas_adata[atlas_adata.obs[celltype_column] == celltype, :].copy()
    print("\n#####\n->", celltype, "\n", adata)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, mask_var="highly_variable")

    X = adata.obsm["X_pca"][:, :10]

    pbar = tqdm(seed_list)
    for seed in pbar:
        pbar.set_description(f"Seed: {seed}")

        X = X.copy()

        n_archetypes = number_of_archetypes_dict[celltype]

        result = time_and_evaluate(
            X, n_runs=N_RUNS, n_archetypes=n_archetypes, seed=seed
        )
        result["seed"] = seed
        result["celltype"] = celltype
        result["n_cells"] = X.shape[0]
        result["n_dim"] = X.shape[1]
        result_list.append(result)

result_df = pd.DataFrame(result_list)
result_df.to_csv(output_dir / "results.csv", index=False)

df_plot = result_df[
    [
        "partipy_time_mean",
        "pypcha_time_mean",
        "partipy_varexpl_mean",
        "pypcha_varexpl_mean",
        "celltype",
        "seed",
    ]
]
df_plot = df_plot.melt(id_vars=["celltype", "seed"])
df_plot["algorithm"] = [v.split("_")[0] for v in df_plot["variable"]]
df_plot["statistic"] = [v.split("_")[1] for v in df_plot["variable"]]
df_plot = df_plot.drop(columns=["variable"])
df_plot = df_plot.pivot(
    index=["celltype", "algorithm", "seed"], columns="statistic", values="value"
)
df_plot = df_plot.reset_index()

color_map = {"partipy": "#10AC5B", "pypcha": "#B01CD1"}

p = (
    pn.ggplot(df_plot)
    + pn.geom_point(pn.aes(x="time", y="varexpl", color="algorithm"), size=5, alpha=0.5)
    + pn.facet_wrap("celltype", scales="free")
    + pn.scale_color_manual(values=color_map)
    + pn.theme_bw()
    + pn.theme(figure_size=(8, 4))
    + pn.labs(x="Time (s)", y="Variance Explained")
)
p.save(figure_dir / "varexpl_vs_time.png", dpi=300, verbose=False)
