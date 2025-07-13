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
from partipy.crosstalk import get_specific_genes_per_archetype, get_archetype_crosstalk, plot_weighted_network

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "hepatocyte_example"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "hepatocyte_example"
output_dir.mkdir(exist_ok=True, parents=True)

# input
data_directory = Path("data")
adata = load_hepatocyte_data_2(data_dir=data_directory)

# preprocessing
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata, mask_var="highly_variable")
adata.layers["z_scaled"]= sc.pp.scale(adata.X, max_value=10, copy=True)

#sc.pl.pca_scatter(adata, color=["Alb", "Cyp2e1", "Apoa2"]) # for later

# number of pc dimensions
pt.compute_shuffled_pca(adata, mask_var="highly_variable")
pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=3)

p = pt.plot_shuffled_pca(adata) + pn.theme_bw()
p.save(figure_dir / "plot_shuffled_pca.pdf")

# archetype selection metrics
pt.compute_selection_metrics(adata=adata, min_k=2, max_k=8)

p = pt.plot_var_explained(adata)
p.save(figure_dir / "plot_var_explained.pdf")

p = pt.plot_IC(adata)
p.save(figure_dir / "plot_IC.pdf")

# bootstrapping
pt.compute_bootstrap_variance(adata=adata, n_bootstrap=50, n_archetypes_list=range(2, 9))

p = pt.plot_bootstrap_variance(adata)
p.save(figure_dir / "plot_bootstrap_variance.pdf")

# compute final archetypes
n_archetypes = 4
pt.compute_archetypes(adata, n_archetypes=n_archetypes, archetypes_only=False)

p = pt.plot_archetypes_2D(adata=adata, show_contours=True) + pn.theme_bw()
p.save(figure_dir / "plot_archetypes_2D_contours.pdf")

p = pt.plot_bootstrap_2D(adata, n_archetypes=4) + pn.theme_bw()
p.save(figure_dir / "plot_bootstrap_2D.pdf")

# archetype characterization
pt.compute_archetype_weights(adata=adata, mode="automatic")
archetype_expression = pt.compute_archetype_expression(adata=adata, layer="z_scaled")

database = "reactome_pathways"
min_genes_per_pathway = 5
max_genes_per_pathway = 80

# dc.op.resource("SIGNOR", organism="mouse")
msigdb_raw = dc.op.resource("MSigDB")
msigdb = msigdb_raw[msigdb_raw["collection"]==database]
selection_vec = ~ np.array(["RESPONSE_TO" in s for s in msigdb["geneset"]]) & \
    ~ np.array(["GENE_EXPRESSION" in s for s in msigdb["geneset"]]) & \
    ~ np.array(["SARS_COV" in s for s in msigdb["geneset"]]) & \
    ~ np.array(["STIMULATED_TRANSCRIPTION" in s for s in msigdb["geneset"]])
msigdb = msigdb.loc[selection_vec, :].copy()
msigdb = msigdb[~msigdb.duplicated(["geneset", "genesymbol"])].copy()
genesets_within_min = (msigdb.value_counts("geneset") >= min_genes_per_pathway).reset_index().query("count")["geneset"].to_list()
genesets_within_max = (msigdb.value_counts("geneset") <= max_genes_per_pathway).reset_index().query("count")["geneset"].to_list()
genesets_to_keep = list(set(genesets_within_min) & set(genesets_within_max))
msigdb = msigdb.loc[msigdb["geneset"].isin(genesets_to_keep), :].copy() # removing small gene sets
msigdb = msigdb.rename(columns={"geneset": "source", "genesymbol": "target"}) # required since decoupler >= 2.0.0

# download might fail because ebi ftp is overloaded...
TRY_DOWNLOAD = False
if TRY_DOWNLOAD:
    msigdb_mouse = dc.op.translate(msigdb, target_organism="mouse") # requires decoupler >= 2.0.0
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

    def _generate_orthologs(
        resource,
        column,
        map_dict,
        one_to_many
    ):
        df = resource[[column]].drop_duplicates().set_index(column)
        df["subunits"] = df.index.str.split("_")
        df["subunits"] = df["subunits"].apply(_replace_subunits, args=(map_dict, one_to_many, ))
        df = df["subunits"].explode().reset_index()
        grouped = df.groupby(column).filter(lambda x: x["subunits"].notna().all()).groupby(column)
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
        result = pd.DataFrame(complexes, columns=col_names).set_index("orthology_source")
        return result

    def _translate(
        resource: pd.DataFrame,
        map_dict: dict,
        column: str,
        one_to_many: int,
    ):  
        map_data = _generate_orthologs(resource, column, map_dict, one_to_many)
        resource = resource.merge(map_data, left_on=column, right_index=True, how="left")
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

    map_df = pd.read_csv(data_directory / "human_mouse_hcop_fifteen_column.txt.gz", sep="\t", low_memory=False)
    map_df['evidence'] = map_df['support'].apply(lambda x: len(x.split(",")))
    map_df = map_df[map_df['evidence'] >= min_evidence]
    map_df = map_df[["human_symbol", target_col]]
    map_df = map_df.rename(columns={'human_symbol': 'source', target_col: 'target'})
    map_dict = map_df.groupby('source')['target'].apply(list).to_dict()

    msigdb_mouse = _translate(resource=msigdb.copy(), map_dict=map_dict, column="target", one_to_many=one_to_many)
    msigdb_mouse = msigdb_mouse.reset_index(drop=True)

msigdb_mouse = msigdb_mouse.drop_duplicates()

acts_ulm_est, acts_ulm_est_p = dc.mt.ulm(data=archetype_expression,
                                         net=msigdb_mouse,
                                         verbose=False)

top_processes = pt.extract_enriched_processes(est=acts_ulm_est, 
                                              pval=acts_ulm_est_p, 
                                              order="desc", 
                                              n=10,
                                              p_threshold=0.05)

for arch_idx in range(n_archetypes):
    adata.obs[f"weights_archetype_{arch_idx}"] = adata.obsm["cell_weights"][:, arch_idx]

    p = pt.plot_archetypes_2D(adata=adata, color=f"weights_archetype_{arch_idx}") + pn.theme_bw()
    p.save(figure_dir / f"plot_archetypes_2D_weights_{arch_idx}.pdf")

    top_processes[arch_idx]["Process"] = pd.Categorical(
        top_processes[arch_idx]["Process"],
        categories=top_processes[arch_idx].sort_values(f"{arch_idx}", ascending=True)["Process"].to_list()
    )

    p = (pn.ggplot(top_processes[arch_idx])
         + pn.geom_col(pn.aes(x="Process", y=f"{arch_idx}"))
         + pn.coord_flip()
         + pn.theme_bw()
         + pn.theme(figure_size=(10, 3))
         )
    p.save(figure_dir / f"plot_top_processes_{arch_idx}.pdf")

# spatial mapping
vizgen_adata = sq.read.vizgen(
    data_directory / "Liver1Slice1",
    counts_file="Liver1Slice1_cell_by_gene.csv",
    meta_file="Liver1Slice1_cell_metadata.csv",
)
vizgen_adata = vizgen_adata[:, vizgen_adata.var.index.isin(archetype_expression.columns.to_list())].copy()
vizgen_adata = vizgen_adata[vizgen_adata.X.sum(axis=1).A1 > 200, :].copy()
sc.pp.normalize_total(vizgen_adata)
sc.pp.log1p(vizgen_adata)

# create weights based on archetype expression profile
archetype_expression_weights = (archetype_expression[vizgen_adata.var.index].copy()
                                .reset_index(names="archetype")
                                .melt(id_vars="archetype", value_name="weight", var_name="gene")
                                )
archetype_expression_weights = archetype_expression_weights.rename(columns={"archetype": "source", "gene": "target"})

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
plotting_df = plotting_df.loc[(plotting_df["x"] > x_low) & (plotting_df["x"] < x_high) & 
                              (plotting_df["y"] > y_low) & (plotting_df["y"] < y_high)]
plotting_df = plotting_df.sample(frac=0.5, random_state=42) # make pdf plots smaller 

plotting_df = plotting_df.melt(id_vars=["x", "y"], value_name="score", var_name="archetype")

p = (pn.ggplot(plotting_df) 
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
specific_genes_per_archetype = get_specific_genes_per_archetype(archetype_expression=archetype_expression, min_score=0.05)

mouse_resource = li.rs.select_resource("mouseconsensus")

archetype_crosstalk = get_archetype_crosstalk(archetype_genes=specific_genes_per_archetype, 
                                              lr_resource=mouse_resource)

fig = plot_weighted_network(
    specific_genes_per_archetype=specific_genes_per_archetype,
    archetype_crosstalk_dict=archetype_crosstalk,
    layout="shell", 
    seed=42, 
    show=False,
    return_fig=True,
    )
fig.savefig(figure_dir / "crosstalk.pdf")
