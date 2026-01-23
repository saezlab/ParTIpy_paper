# manual download from here: https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy
# see Leo's analysis here: https://github.com/saezlab/best_practices_ParTIpy/tree/main
from pathlib import Path

import plotnine as pn
import scanpy as sc
import partipy as pt

from ..utils.const import FIGURE_PATH, OUTPUT_PATH, DATA_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "fibroblast_cross_condition"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "fibroblast_cross_condition"
output_dir.mkdir(exist_ok=True, parents=True)

adata = sc.read_h5ad(Path(DATA_PATH) / "human_dcm_hcm_scportal_03.17.2022.h5ad")
adata = adata[
    adata.obs["cell_type_leiden0.6"].isin(
        ["Fibroblast_I", "Activated_fibroblast", "Fibroblast_II"]
    ),
    :,
].copy()

min_genes_per_cell = 100
min_cells_per_gene = 10

sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

adata.obs["disease_original"] = adata.obs["disease"].copy()
adata.obs["disease"] = adata.obs["disease_original"].map(
    {"HCM": "CM", "DCM": "CM", "NF": "NF"}
)

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata, mask_var="highly_variable")
adata.layers["z_scaled"] = sc.pp.scale(adata.X, max_value=10)

print(adata.obs["disease"].value_counts())

pt.compute_shuffled_pca(adata, mask_var="highly_variable", n_shuffle=25)
p = pt.plot_shuffled_pca(adata) + pn.theme_bw()
p.save(figure_dir / "plot_shuffled_pca.pdf")

pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=16)

pt.compute_selection_metrics(adata=adata, n_archetypes_list=range(2, 8))

p = pt.plot_var_explained(adata)
p.save(figure_dir / "plot_var_explained.pdf")

p = pt.plot_IC(adata)
p.save(figure_dir / "plot_IC.pdf")


pt.compute_bootstrap_variance(
    adata=adata, n_bootstrap=50, n_archetypes_list=range(2, 8)
)

p = pt.plot_bootstrap_variance(adata)
p.save(figure_dir / "plot_bootstrap_variance.pdf")

p = pt.plot_archetypes_2D(
    adata=adata, show_contours=True, result_filters={"n_archetypes": 4}, alpha=0.05
)
p.save(figure_dir / "plot_archetypes_2D.pdf")

p = pt.plot_bootstrap_2D(adata, result_filters={"n_archetypes": 4})
p.save(figure_dir / "plot_bootstrap_2D.pdf")

p = pt.plot_archetypes_2D(
    adata=adata,
    show_contours=True,
    color="disease",
    alpha=0.05,
    size=0.5,
    result_filters={"n_archetypes": 4},
) + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1.0, "size": 4.0}))
p.save(figure_dir / "plot_archetypes_2D_disease.pdf")

pt.compute_archetype_weights(adata=adata, mode="automatic", result_filters={"n_archetypes": 4})
disease_enrichment = pt.compute_meta_enrichment(adata=adata, meta_col="disease", result_filters={"n_archetypes": 4})
p = pt.barplot_meta_enrichment(disease_enrichment, meta="disease")
p.save(figure_dir / "barplot_meta_enrichment_disease.pdf")

disease_orginal_enrichment = pt.compute_meta_enrichment(adata=adata, meta_col="disease_original", result_filters={"n_archetypes": 4})
p = pt.barplot_meta_enrichment(disease_orginal_enrichment, meta="disease_original")
p.save(figure_dir / "barplot_meta_enrichment_disease_original.pdf")

# save this for now on sds and locally
pt.write_h5ad(adata, output_dir / "fibroblast_cross_condition_partipy.h5ad")
pt.write_h5ad(adata, "/home/pschaefer/fibroblast_cross_condition_partipy.h5ad")
