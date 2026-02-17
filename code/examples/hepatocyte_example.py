from pathlib import Path

import scanpy as sc
import pandas as pd
import numpy as np
import decoupler as dc
import plotnine as pn
import squidpy as sq
import liana as li
import partipy as pt
from partipy.datasets import load_hepatocyte_data_2
from partipy.crosstalk import (
    get_specific_genes_per_archetype,
    get_archetype_crosstalk,
    plot_weighted_network,
)
from scipy.stats import pearsonr
from partipy.paretoti import summarize_aa_metrics

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "hepatocyte_example"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "hepatocyte_example"
output_dir.mkdir(exist_ok=True, parents=True)

source_data = Path(OUTPUT_PATH) / "source_data"
source_data.mkdir(exist_ok=True, parents=True)

# input
data_directory = Path("data")
adata = load_hepatocyte_data_2(data_dir=data_directory)

# preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata, mask_var="highly_variable")
adata.layers["z_scaled"] = sc.pp.scale(adata.X, max_value=10, copy=True)

# sc.pl.pca_scatter(adata, color=["Alb", "Cyp2e1", "Apoa2"]) # for later

# number of pc dimensions
pt.compute_shuffled_pca(adata, mask_var="highly_variable")
pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=3)

p = pt.plot_shuffled_pca(adata) + pn.theme_bw()
p.save(figure_dir / "plot_shuffled_pca.pdf")

# archetype selection metrics
pt.compute_selection_metrics(adata=adata, min_k=2, max_k=8)

plot_df = summarize_aa_metrics(adata)
plot_df.to_csv(source_data / "figure_3_panel_A.csv", index=False)
plot_df.to_csv(source_data / "figure_3_panel_B.csv", index=False)
plot_df.to_csv(source_data / "EV_1_panel_A.csv", index=False)

p = pt.plot_var_explained(adata)
p.save(figure_dir / "plot_var_explained.pdf")

p = pt.plot_IC(adata)
p.save(figure_dir / "plot_IC.pdf")

# bootstrapping
pt.compute_bootstrap_variance(
    adata=adata, n_bootstrap=50, n_archetypes_list=range(2, 9)
)

pd.DataFrame(adata.obsm["X_pca"][:, :2], columns=["X_pca_0", "X_pca_1"]).to_csv(
    source_data / "figure_3_panel_C_part_1.csv", index=False
)
pt.get_aa_bootstrap(adata, n_archetypes=4).to_csv(
    source_data / "figure_3_panel_C_part_2.csv", index=False
)
pt.get_aa_bootstrap(adata, n_archetypes=4).to_csv(
    source_data / "EV_1_panel_B.csv", index=False
)

p = pt.plot_bootstrap_variance(adata)
p.save(figure_dir / "plot_bootstrap_variance.pdf")

n_archetypes = 4

p = (
    pt.plot_archetypes_2D(
        adata=adata, show_contours=True, result_filters={"n_archetypes": n_archetypes}
    )
    + pn.theme_bw()
)
p.save(figure_dir / "plot_archetypes_2D_contours.pdf")

p = (
    pt.plot_bootstrap_2D(adata, result_filters={"n_archetypes": n_archetypes})
    + pn.theme_bw()
)
p.save(figure_dir / "plot_bootstrap_2D.pdf")

# archetype characterization
pt.compute_archetype_weights(
    adata=adata, mode="automatic", result_filters={"n_archetypes": n_archetypes}
)
archetype_expression = pt.compute_archetype_expression(
    adata=adata,
    layer="z_scaled",
    result_filters={"n_archetypes": n_archetypes},
)

database = "reactome_pathways"
min_genes_per_pathway = 5
max_genes_per_pathway = 80

# dc.op.resource("SIGNOR", organism="mouse")
msigdb_raw = dc.op.resource("MSigDB")
msigdb = msigdb_raw[msigdb_raw["collection"] == database]
selection_vec = (
    ~np.array(["RESPONSE_TO" in s for s in msigdb["geneset"]])
    & ~np.array(["GENE_EXPRESSION" in s for s in msigdb["geneset"]])
    & ~np.array(["SARS_COV" in s for s in msigdb["geneset"]])
    & ~np.array(["STIMULATED_TRANSCRIPTION" in s for s in msigdb["geneset"]])
)
msigdb = msigdb.loc[selection_vec, :].copy()
msigdb = msigdb[~msigdb.duplicated(["geneset", "genesymbol"])].copy()
genesets_within_min = (
    (msigdb.value_counts("geneset") >= min_genes_per_pathway)
    .reset_index()
    .query("count")["geneset"]
    .to_list()
)
genesets_within_max = (
    (msigdb.value_counts("geneset") <= max_genes_per_pathway)
    .reset_index()
    .query("count")["geneset"]
    .to_list()
)
genesets_to_keep = list(set(genesets_within_min) & set(genesets_within_max))
msigdb = msigdb.loc[
    msigdb["geneset"].isin(genesets_to_keep), :
].copy()  # removing small gene sets
msigdb = msigdb.rename(
    columns={"geneset": "source", "genesymbol": "target"}
)  # required since decoupler >= 2.0.0

# download might fail because ebi ftp is overloaded...
TRY_DOWNLOAD = False
if TRY_DOWNLOAD:
    msigdb_mouse = dc.op.translate(
        msigdb, target_organism="mouse"
    )  # requires decoupler >= 2.0.0
else:
    # helpers
    from itertools import product

    def _replace_subunits(
        lst: list,
        my_dict: dict,
        one_to_many: int,
    ):
        result = []
        for x in lst:
            if x in my_dict:
                value = my_dict[x]
                if not isinstance(value, list):
                    value = [value]
                if len(value) > one_to_many:
                    result.append(np.nan)
                else:
                    result.append(value)
            else:
                result.append(np.nan)
        return result

    def _generate_orthologs(resource, column, map_dict, one_to_many):
        df = resource[[column]].drop_duplicates().set_index(column)
        df["subunits"] = df.index.str.split("_")
        df["subunits"] = df["subunits"].apply(
            _replace_subunits,
            args=(
                map_dict,
                one_to_many,
            ),
        )
        df = df["subunits"].explode().reset_index()
        grouped = (
            df.groupby(column)
            .filter(lambda x: x["subunits"].notna().all())
            .groupby(column)
        )
        # Generate all possible subunit combinations within each group
        complexes = []
        for name, group in grouped:
            if group["subunits"].isnull().all():
                continue
            subunit_lists = [list(x) for x in group["subunits"]]
            complex_combinations = list(product(*subunit_lists))
            for complex in complex_combinations:
                complexes.append((name, "_".join(complex)))
        # Create output DataFrame
        col_names = ["orthology_source", "orthology_target"]
        result = pd.DataFrame(complexes, columns=col_names).set_index(
            "orthology_source"
        )
        return result

    def _translate(
        resource: pd.DataFrame,
        map_dict: dict,
        column: str,
        one_to_many: int,
    ):
        map_data = _generate_orthologs(resource, column, map_dict, one_to_many)
        resource = resource.merge(
            map_data, left_on=column, right_index=True, how="left"
        )
        resource[column] = resource["orthology_target"]
        resource = resource.drop(columns=["orthology_target"])
        resource = resource.dropna(subset=[column])
        return resource

    # default params
    target_organism = "mouse"
    columns = ["source", "target", "genesymbol"]
    min_evidence = 3
    one_to_many = 5
    target_col = f"{target_organism}_symbol"

    map_df = pd.read_csv(
        data_directory / "human_mouse_hcop_fifteen_column.txt.gz",
        sep="\t",
        low_memory=False,
    )
    map_df["evidence"] = map_df["support"].apply(lambda x: len(x.split(",")))
    map_df = map_df[map_df["evidence"] >= min_evidence]
    map_df = map_df[["human_symbol", target_col]]
    map_df = map_df.rename(columns={"human_symbol": "source", target_col: "target"})
    map_dict = map_df.groupby("source")["target"].apply(list).to_dict()

    msigdb_mouse = _translate(
        resource=msigdb.copy(),
        map_dict=map_dict,
        column="target",
        one_to_many=one_to_many,
    )
    msigdb_mouse = msigdb_mouse.reset_index(drop=True)

msigdb_mouse = msigdb_mouse.drop_duplicates()

acts_ulm_est, acts_ulm_est_p = dc.mt.ulm(
    data=archetype_expression, net=msigdb_mouse, verbose=False
)

top_processes = pt.extract_enriched_processes(
    est=acts_ulm_est, pval=acts_ulm_est_p, order="desc", n=10, p_threshold=0.05
)

df_panel_D_df = pd.DataFrame(adata.obsm["X_pca"][:, :2], columns=["X_pca_0", "X_pca_1"])
df_panel_E_df = []

for arch_idx in range(n_archetypes):
    adata.obs[f"weights_archetype_{arch_idx}"] = pt.get_aa_cell_weights(
        adata, n_archetypes=n_archetypes
    )[:, arch_idx]

    df_panel_D_df[f"weights_archetype_{arch_idx}"] = pt.get_aa_cell_weights(
        adata, n_archetypes=n_archetypes
    )[:, arch_idx]

    p = (
        pt.plot_archetypes_2D(
            adata=adata,
            color=f"weights_archetype_{arch_idx}",
            result_filters={"n_archetypes": n_archetypes},
        )
        + pn.theme_bw()
    )
    p.save(figure_dir / f"plot_archetypes_2D_weights_{arch_idx}.pdf")

    top_processes[arch_idx][f"neg_log10_pval_{arch_idx}"] = -np.log10(
        top_processes[arch_idx][f"pval_{arch_idx}"]
    )
    top_processes[arch_idx]["Process"] = top_processes[arch_idx]["Process"].str.replace(
        "REACTOME_", ""
    )

    top_processes[arch_idx]["Process"] = pd.Categorical(
        top_processes[arch_idx]["Process"],
        categories=top_processes[arch_idx]
        .sort_values(f"{arch_idx}", ascending=True)["Process"]
        .to_list(),
    )

    df_panel_E_df.append(top_processes[arch_idx])

    p = (
        pn.ggplot(top_processes[arch_idx])
        + pn.geom_col(
            pn.aes(x="Process", y=f"act_{arch_idx}", fill=f"neg_log10_pval_{arch_idx}")
        )
        + pn.coord_flip()
        + pn.theme_bw()
        + pn.theme(figure_size=(9, 3))
        + pn.labs(x="", y=f"t-value in Archetype {arch_idx}", fill="$-\log10(P_{adj})$")
        + pn.scale_fill_gradient(low="#c8f9b9", high="#006d2c")
    )
    p.save(figure_dir / f"plot_top_processes_{arch_idx}.pdf")

df_panel_D_df.to_csv(source_data / "figure_3_panel_D.csv", index=False)
df_panel_D_df.to_csv(source_data / "EV_1_panel_D.csv", index=False)
pd.concat(df_panel_E_df).to_csv(source_data / "figure_3_panel_E.csv", index=False)
pd.concat(df_panel_E_df).to_csv(source_data / "EV_1_panel_E.csv", index=False)

# spatial mapping
vizgen_adata = sq.read.vizgen(
    data_directory / "Liver1Slice1",
    counts_file="Liver1Slice1_cell_by_gene.csv",
    meta_file="Liver1Slice1_cell_metadata.csv",
)
vizgen_adata = vizgen_adata[
    :, vizgen_adata.var.index.isin(archetype_expression.columns.to_list())
].copy()
vizgen_adata = vizgen_adata[vizgen_adata.X.sum(axis=1).A1 > 200, :].copy()
sc.pp.normalize_total(vizgen_adata)
sc.pp.log1p(vizgen_adata)

# create weights based on archetype expression profile
archetype_expression_weights = (
    archetype_expression[vizgen_adata.var.index]
    .copy()
    .reset_index(names="archetype")
    .melt(id_vars="archetype", value_name="weight", var_name="gene")
)
archetype_expression_weights = archetype_expression_weights.rename(
    columns={"archetype": "source", "gene": "target"}
)

# compute the enrichment of the archetype gex profiles in the gex profiles of each cell
dc.mt.ulm(data=vizgen_adata, net=archetype_expression_weights, verbose=False)

# plotting one part
plotting_df = vizgen_adata.obsm["score_ulm"].copy()
plotting_df.columns = ["archetype_" + idx for idx in plotting_df.columns]
plotting_df = plotting_df - plotting_df.mean()
plotting_df = plotting_df / plotting_df.std()
plotting_df["x"] = vizgen_adata.obsm["spatial"][:, 0]
plotting_df["y"] = vizgen_adata.obsm["spatial"][:, 1]

width = 3_000
x_low = 5_000
y_low = 6_000
x_high = x_low + width
y_high = y_low + width
plotting_df = plotting_df.loc[
    (plotting_df["x"] > x_low)
    & (plotting_df["x"] < x_high)
    & (plotting_df["y"] > y_low)
    & (plotting_df["y"] < y_high)
]
plotting_df = plotting_df.sample(frac=0.5, random_state=42)  # make pdf plots smaller

plotting_df = plotting_df.melt(
    id_vars=["x", "y"], value_name="score", var_name="archetype"
)
plotting_df.to_csv(source_data / "figure_3_panel_F.csv", index=False)
plotting_df.to_csv(source_data / "EV_1_panel_F.csv", index=False)

p = (
    pn.ggplot(plotting_df)
    + pn.geom_point(pn.aes(x="x", y="y", color="score"), size=0.25, alpha=0.5)
    + pn.scale_color_gradient2(low="blue", mid="white", high="red", midpoint=0)
    + pn.facet_wrap(facets="archetype", nrow=2, ncol=2)
    + pn.coord_equal()
    + pn.theme_bw()
    + pn.theme(figure_size=(8, 8))
    + pn.labs(x="x", y="y", color="Activity")
)
p.save(figure_dir / "spatial_mapping.pdf")

# archetype crosstalk network
specific_genes_per_archetype = get_specific_genes_per_archetype(
    archetype_expression=archetype_expression, min_score=0.05
)

mouse_resource = li.rs.select_resource("mouseconsensus")

archetype_crosstalk = get_archetype_crosstalk(
    archetype_genes=specific_genes_per_archetype, lr_resource=mouse_resource
)

flat_df = pd.concat(
    [
        df.assign(sender_archetype=sender_arch, receiver_archetype=receiver_arch)
        for sender_arch, receivers in archetype_crosstalk.items()
        for receiver_arch, df in receivers.items()
    ],
    ignore_index=True,
)
flat_df.to_csv(source_data / "EV_1_panel_C.csv", index=False)

fig = plot_weighted_network(
    specific_genes_per_archetype=specific_genes_per_archetype,
    archetype_crosstalk_dict=archetype_crosstalk,
    layout="shell",
    seed=42,
    show=False,
    return_fig=True,
)
fig.savefig(figure_dir / "crosstalk.pdf")

########################################################################
# Save final adata object
########################################################################
pt.write_h5ad(adata, output_dir / "hepatocyte.h5ad")

########################################################################
# Characterize archetypal gene expression
########################################################################
arch_expr = {
    "log1p": pt.compute_archetype_expression(
        adata=adata, layer=None
    ),  # assumes log1p in .X
    "z_scaled": pt.compute_archetype_expression(adata=adata, layer="z_scaled"),
}

long_parts = []
for scale, df_wide in arch_expr.items():
    part = (
        df_wide.rename_axis(index="archetype")  # name index
        .reset_index()  # archetype column
        .melt(id_vars="archetype", var_name="gene", value_name="value")
    )
    part["archetype"] = part["archetype"].map(lambda a: f"arch_{a}")  # optional
    part["scale"] = scale
    long_parts.append(part)

arch_expr_long = pd.concat(long_parts, ignore_index=True)

arch_expr_long = arch_expr_long.pivot_table(
    values="value", index=["archetype", "gene"], columns="scale"
).reset_index()
arch_expr_long.columns.name = None

# Optional: stable column order
col_order = ["archetype", "gene", "raw", "log1p", "z_scaled"]
arch_expr_long = arch_expr_long.reindex(
    columns=[c for c in col_order if c in arch_expr_long.columns]
)
arch_expr_long.to_csv(output_dir / "archetype_expression_pivot.csv", index=False)

########################################################################
# Compute gene enrichment per archetype
########################################################################
enrichment_df = pt.compute_quantile_based_gene_enrichment(
    adata, result_filters={"n_archetypes": 4}, verbose=True
)
enrichment_df.to_csv(output_dir / "enrichment_df.csv", index=False)

########################################################################
# Compare quantile-based enrichment vs. kernel aggregation
########################################################################
arch_expr_long_copy = arch_expr_long.copy()
arch_expr_long_copy["arch_idx"] = (
    arch_expr_long_copy["archetype"].str.extract(r"arch_([0-9]+)").astype(int)
)
arch_expr_long_copy = arch_expr_long_copy.drop(columns=["archetype"])

comparison_df = enrichment_df.join(
    arch_expr_long_copy.set_index(["arch_idx", "gene"]),
    on=["arch_idx", "gene"],
    how="left",
)
comparison_df.to_csv(output_dir / "comparison_df.csv", index=False)


def helper_compute_corr(df):
    r, p = pearsonr(df["z_scaled"], df["stat"])
    return pd.Series({"r": r, "p": p, "label": f"r = {r:.2f}\np = {p:.2e}"})


corr_df = comparison_df.groupby("arch_idx", as_index=False).apply(helper_compute_corr)

pos_df = (
    comparison_df.groupby("arch_idx")
    .agg(x=("z_scaled", "min"), y=("stat", "max"))
    .reset_index()
)

corr_df = corr_df.merge(pos_df, on="arch_idx")

p = (
    pn.ggplot(comparison_df)
    + pn.geom_point(
        pn.aes(x="z_scaled", y="stat", color="enriched"), alpha=0.5, size=1.0
    )
    + pn.geom_smooth(
        pn.aes(x="z_scaled", y="stat"), method="lm", alpha=0.5, color="grey"
    )
    + pn.geom_text(
        corr_df, pn.aes(x="x", y="y", label="label"), ha="left", va="top", size=8
    )
    + pn.facet_wrap("arch_idx", ncol=4)
    + pn.labs(x="z-scored Gene Expression", y="Test Statistic", color="Enriched")
    + pn.theme_bw()
    + pn.theme(
        figure_size=(12, 3),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
        strip_background=pn.element_rect(fill="white", color="black"),
        strip_text=pn.element_text(color="black"),
    )
    + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1, "size": 3}))
)
p.save(figure_dir / "enrichment_comparison_scatter_plot.pdf", verbose=False)
p.save(figure_dir / "enrichment_comparison_scatter_plot.png", verbose=False, dpi=300)
