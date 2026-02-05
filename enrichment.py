from pathlib import Path

import scanpy as sc
import partipy as pt



figure_dir = Path("figures") / "fibroblast_cross_condition"
figure_dir.mkdir(exist_ok=True, parents=True)
sc.settings.figdir = figure_dir

output_dir = Path("output") / "fibroblast_cross_condition"
output_dir.mkdir(exist_ok=True, parents=True)

n_archetypes = 3
obsm_key = "X_pca_harmony"
obsm_dim = 16

adata = pt.read_h5ad("/home/pschaefer/fibroblast_cross_condition_partipy.h5ad")

gene_mask = adata.var.index[adata.var["n_cells"] > 100].to_list()
adata = adata[:, gene_mask].copy()

enrichment_df = pt.compute_quantile_based_gene_enrichment(
    adata, result_filters={"n_archetypes": 4}, verbose=True
)

enrichment_df.to_csv("enrichment_df.csv")
