from pathlib import Path
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import scanpy as sc
import partipy as pt
from partipy.utils import align_archetypes, compute_relative_rowwise_l2_distance
import plotnine as pn
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Gaussian
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..utils.data_utils import load_ms_data
from ..utils.const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

## define helper function
def get_minimal_value_key(dict_input):
    return int(
        np.array(list(dict_input.keys()))[
            np.argmin(np.array(list(dict_input.values())))
        ]
    )

def pearsonr_per_row(mtx_1: np.ndarray, mtx_2: np.ndarray, return_pval: bool = False):
    from scipy.stats import pearsonr
    assert mtx_1.shape == mtx_2.shape
    corr_list = []
    pval_list = []
    for row_idx in range(mtx_1.shape[0]):
        pearson_out = pearsonr(mtx_1[row_idx, :], mtx_2[row_idx, :])
        corr_list.append(pearson_out.statistic)
        pval_list.append(pearson_out.pvalue)
    if return_pval:
        return np.array(corr_list), np.array(pval_list)
    else:
        return np.array(corr_list)

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_coreset"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_coreset"
output_dir.mkdir(exist_ok=True, parents=True)

## setting up different seeds to test TODO: Change this
seed_list = SEED_DICT["m"]

coreset_fraction_list = 1 / (np.array([2**n for n in range(0, 8)]) * (25 / 16))
coreset_fraction_arr = np.zeros(len(coreset_fraction_list) + 1)
coreset_fraction_arr[1:] = coreset_fraction_list
coreset_fraction_arr[0] = 1.0

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
archetypes_to_test = list(range(2, 10))
number_of_archetypes_dict = {
    "MG": 6,
    "AS": 5,
    "OL": 5,
    "OPC": 4,
    "NEU": 6,
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
    adata = atlas_adata[atlas_adata.obs[celltype_column] == celltype, :].copy()
    print("\n#####\n->", celltype, "\n", adata)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, mask_var="highly_variable")
    adata.layers["z_scaled"] = sc.pp.scale(adata.X, max_value=10, copy=True)

    ## some scanpy QC plots
    for qc_var in qc_columns:
        adata.obs[qc_var] = pd.Categorical(adata.obs[qc_var])
    sc.pl.highly_variable_genes(adata, log=False, show=False, save=False)
    plt.savefig(figure_dir_celltype / "highly_variable_genes.png")
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=False, show=False, save=False)
    plt.savefig(figure_dir_celltype / "pca_var_explained.png")
    sc.pl.pca(
        adata,
        color=qc_columns,
        dimensions=[(0, 1), (0, 1)],
        ncols=2,
        size=10,
        alpha=0.75,
        show=False,
        save=False,
    )
    plt.savefig(figure_dir_celltype / "pca_2D.png")

    ## for simplicity we will always use 10 principal components
    pt.set_obsm(
        adata=adata, obsm_key="X_pca", n_dimensions=number_of_pcs_dict[celltype]
    )

    # reference archetype
    adata_ref = adata.copy()
    pt.compute_archetypes(
        adata_ref,
        n_archetypes=number_of_archetypes_dict[celltype],
        seed=42,
        save_to_anndata=True,
        archetypes_only=False,
        verbose=False,
    )

    reference_archetypes_pos_dict = {}
    reference_archetype_char_gex_dict = {}

    for coreset_fraction in coreset_fraction_arr:
        print(coreset_fraction)

        pbar = tqdm(seed_list)
        for seed in pbar:
            pbar.set_description(f"Seed: {seed}")

            adata_bench = adata.copy()

            start_time = time.time()

            pt.compute_archetypes(
                adata_bench,
                n_archetypes=number_of_archetypes_dict[celltype],
                n_restarts=1,
                coreset_algorithm="standard" if coreset_fraction < 1 else None,
                coreset_fraction=coreset_fraction,
                seed=seed,
                save_to_anndata=True,
                archetypes_only=False,
                verbose=False,
            )

            # compute gene enrichment
            pt.compute_archetype_weights(adata=adata_bench, mode="automatic")
            archetype_expression = pt.compute_archetype_expression(
                adata=adata_bench, layer="z_scaled"
            )

            if coreset_fraction == 1.0:
                reference_archetypes_pos_dict[seed] = adata_bench.uns["AA_results"]["Z"]
                reference_archetype_char_gex_dict[seed] = archetype_expression

            end_time = time.time()
            execution_time = end_time - start_time

            # compute euclidean distance to all reference archetype runs
            euclidean_distances = {}
            query_idx_dict = {}
            for seed_key, ref_arch in reference_archetypes_pos_dict.items():
                query_arch = adata_bench.uns["AA_results"]["Z"].copy()
                euclidean_d = cdist(ref_arch, query_arch, metric="euclidean")

                # Find the optimal assignment using the Hungarian algorithm
                _ref_idx, query_idx = linear_sum_assignment(euclidean_d)

                # compute mean euclidean distance
                euclidean_distance_per_matched_arch = np.sum(
                    (ref_arch[_ref_idx, :] - query_arch[query_idx, :]) ** 2, axis=1
                )
                euclidean_distances[seed_key] = np.mean(
                    euclidean_distance_per_matched_arch
                )
                query_idx_dict[seed_key] = query_idx

            # compute mean pearson correlation of the pathway enrichment
            seed_min = get_minimal_value_key(euclidean_distances)
            pearson_corr_per_matched_arch = pearsonr_per_row(
                reference_archetype_char_gex_dict[seed_min].to_numpy(),
                archetype_expression.iloc[query_idx_dict[seed_min], :].to_numpy()
            )

            Z = adata_ref.uns["AA_results"]["Z"].copy()
            Z_hat = adata_bench.uns["AA_results"]["Z"].copy()
            Z_hat = align_archetypes(Z, Z_hat)
            rel_dist_between_archetypes = compute_relative_rowwise_l2_distance(Z, Z_hat)

            result_dict = {
                "celltype": celltype,
                "time": execution_time,
                "min_l2_distance_to_ref": euclidean_distances[seed_min],
                "mean_gex_corr": np.mean(pearson_corr_per_matched_arch),
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

ncols = 4
n_celltypes = result_df["celltype"].nunique()
nrows = int(np.ceil(n_celltypes / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
axes = axes.flatten()

min_coresets_list = []

for (celltype, df_group), ax in zip(result_df.groupby("celltype"), axes):
    df_group = df_group.copy()

    n_samples = df_group["n_samples"].to_numpy()[0]
    n_dimensions = df_group["n_dimensions"].to_numpy()[0]
    n_archetypes = df_group["n_archetypes"].to_numpy()[0]

    df_group["log10_coreset_fraction"] = np.log10(df_group["coreset_fraction"])
    top_varexpl = df_group.loc[df_group["coreset_fraction"] == 1.0]["varexpl"].mean()
    deviation_fraction = 0.01
    target_varexpl = top_varexpl - (top_varexpl * deviation_fraction)
    df_group = df_group[["log10_coreset_fraction", "varexpl"]].dropna()
    df_group = df_group.sort_values("log10_coreset_fraction")

    # extract the variables
    x = df_group["log10_coreset_fraction"].values
    y = df_group["varexpl"].values

    # Create B-spline basis functions
    # You can adjust the number of knots and degree as needed
    n_splines = 7  # Number of basis functions
    spline_basis = BSplines(x, df=n_splines, degree=2)

    # Add intercept term
    intercept = np.ones((len(x), 1))

    # Fit the GAM model with intercept
    gam_model = GLMGam(y, exog=intercept, smoother=spline_basis, family=Gaussian())
    gam_results = gam_model.fit()

    # Generate predictions for plotting
    x_pred = np.linspace(x.min(), x.max(), 100)

    # Create intercept for predictions
    intercept_pred = np.ones((len(x_pred), 1))

    # Get predictions using the model"s predict method
    y_pred_mean = gam_results.predict(
        exog=intercept_pred, exog_smooth=x_pred.reshape(-1, 1)
    )

    # For confidence intervals, we"ll use a simpler approach
    # Calculate standard errors manually or use bootstrap if needed
    # For now, let"s create a basic confidence interval using residual std
    residual_std = np.std(gam_results.resid_response)
    y_pred_ci = 1.96 * residual_std  # Approximate 95% CI

    # Residual analysis
    residuals = gam_results.resid_response
    fitted_values = gam_results.fittedvalues

    # some more things
    min_corset_size = x_pred[y_pred_mean > target_varexpl].min()
    min_varexpl = y_pred_mean[y_pred_mean > target_varexpl].min()

    min_coresets_list.append(
        {
            "celltype": celltype,
            "log10_coreset_size": min_corset_size,
            "coreset_size": 10**min_corset_size,
            "varexpl": min_varexpl,
        }
    )

    # Plot the results
    ax.vlines(
        x=0,
        ymin=top_varexpl - (top_varexpl * deviation_fraction),
        ymax=top_varexpl + (top_varexpl * deviation_fraction),
        color="black",
    )
    ax.scatter(x, y, alpha=0.5, label="Data points")
    ax.plot(x_pred, y_pred_mean, "r-", linewidth=2, label="GAM fit")
    ax.fill_between(
        x_pred,
        y_pred_mean - y_pred_ci,
        y_pred_mean + y_pred_ci,
        alpha=0.3,
        color="red",
        label="Approx 95% CI",
    )
    ax.axvline(min_corset_size, color="green")
    ax.axhline(min_varexpl, color="green")
    ax.set_title(
        f"{celltype} | {n_samples} | min corset fraction: {10**min_corset_size:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.savefig(figure_dir / "varexpl_vs_coreset_fraction_gam.png")
plt.close()

min_corset_df = pd.DataFrame(min_coresets_list)
min_corset_df.to_csv(output_dir / "min_coresets.csv", index=False)

ncols = 4
n_celltypes = result_df["celltype"].nunique()
nrows = int(np.ceil(n_celltypes / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
axes = axes.flatten()

time_savings_list = []

for (celltype, df_group), ax in zip(result_df.groupby("celltype"), axes):
    df_group = df_group.copy()

    n_samples = df_group["n_samples"].to_numpy()[0]
    n_dimensions = df_group["n_dimensions"].to_numpy()[0]
    n_archetypes = df_group["n_archetypes"].to_numpy()[0]

    df_group["log10_coreset_fraction"] = np.log10(df_group["coreset_fraction"])
    fulltime = df_group.loc[df_group["coreset_fraction"] == 1.0]["time"].mean()
    df_group = df_group[["log10_coreset_fraction", "time"]].dropna()
    df_group = df_group.sort_values("log10_coreset_fraction")

    # extract the variables
    x = df_group["log10_coreset_fraction"].values
    y = df_group["time"].values

    # Create B-spline basis functions
    # You can adjust the number of knots and degree as needed
    n_splines = 7  # Number of basis functions
    spline_basis = BSplines(x, df=n_splines, degree=2)

    # Add intercept term
    intercept = np.ones((len(x), 1))

    # Fit the GAM model with intercept
    gam_model = GLMGam(y, exog=intercept, smoother=spline_basis, family=Gaussian())
    gam_results = gam_model.fit()

    # Generate predictions for plotting
    x_pred = np.linspace(x.min(), x.max(), 100)

    # Create intercept for predictions
    intercept_pred = np.ones((len(x_pred), 1))

    # Get predictions using the model"s predict method
    y_pred_mean = gam_results.predict(
        exog=intercept_pred, exog_smooth=x_pred.reshape(-1, 1)
    )

    # For confidence intervals, we"ll use a simpler approach
    # Calculate standard errors manually or use bootstrap if needed
    # For now, let"s create a basic confidence interval using residual std
    residual_std = np.std(gam_results.resid_response)
    y_pred_ci = 1.96 * residual_std  # Approximate 95% CI

    # Residual analysis
    residuals = gam_results.resid_response
    fitted_values = gam_results.fittedvalues

    # some more things
    min_corset_size = min_corset_df.set_index("celltype").loc[celltype][
        "log10_coreset_size"
    ]
    idx = int(np.argmin((x_pred - min_corset_size) ** 2))
    min_time = y_pred_mean[idx]
    time_saving = np.round(fulltime / min_time, 1)

    time_savings_list.append(
        {
            "celltype": celltype,
            "log10_corset_size": min_corset_size,
            "coreset_size": 10**min_corset_size,
            "coreset_time": min_time,
            "full_time": fulltime,
            "time_saving": time_saving,
        }
    )

    # Plot the results
    ax.scatter(x, y, alpha=0.5, label="Data points")
    ax.plot(x_pred, y_pred_mean, "r-", linewidth=2, label="GAM fit")
    ax.fill_between(
        x_pred,
        y_pred_mean - y_pred_ci,
        y_pred_mean + y_pred_ci,
        alpha=0.3,
        color="red",
        label="Approx 95% CI",
    )
    ax.axvline(min_corset_size, color="green")
    ax.axhline(min_time, color="green")
    ax.set_title(f"{celltype} | {n_samples} | time saving: {time_saving:.1f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.savefig(figure_dir / "time_vs_coreset_fraction_gam.png")
plt.close()

time_savings_df = pd.DataFrame(time_savings_list)
time_savings_df.to_csv(output_dir / "time_savings.csv", index=False)

# lastly some ggplots
p = (
    pn.ggplot(result_df)
    + pn.geom_point(pn.aes(x="coreset_fraction", y="time"))
    + pn.geom_smooth(pn.aes(x="coreset_fraction", y="time"), method="loess")
    + pn.geom_vline(data=min_corset_df, mapping=pn.aes(xintercept="coreset_size"))
    + pn.facet_wrap(facets="celltype", scales="free_y", ncol=4)
    + pn.theme(figure_size=(10, 5))
    + pn.scale_x_log10()
    + pn.ylim((0, None))
)
p.save(figure_dir / "time_vs_coreset_fraction.png", dpi=300, verbose=False)

p = (
    pn.ggplot(result_df)
    + pn.geom_point(pn.aes(x="coreset_fraction", y="mean_rel_l2_distance"))
    + pn.geom_smooth(
        pn.aes(x="coreset_fraction", y="mean_rel_l2_distance"), method="loess"
    )
    + pn.geom_vline(data=min_corset_df, mapping=pn.aes(xintercept="coreset_size"))
    + pn.facet_wrap(facets="celltype", scales="free_y", ncol=4)
    + pn.theme(figure_size=(10, 5))
    + pn.scale_x_log10()
    + pn.ylim((0, None))
)
p.save(
    figure_dir / "mean_rel_l2_distance_vs_coreset_fraction.png", dpi=300, verbose=False
)

p = (
    pn.ggplot(result_df)
    + pn.geom_point(pn.aes(x="coreset_fraction", y="varexpl"))
    + pn.geom_smooth(pn.aes(x="coreset_fraction", y="varexpl"), method="loess")
    + pn.geom_vline(data=min_corset_df, mapping=pn.aes(xintercept="coreset_size"))
    + pn.facet_wrap(facets="celltype", scales="free_y", ncol=4)
    + pn.theme(figure_size=(10, 5))
    + pn.scale_x_log10()
)
p.save(figure_dir / "varexpl_vs_coreset_fraction.png", dpi=300, verbose=False)
