# Fibroblast Cross-Condition Analysis Summary

## Overview
This analysis compares human cardiac fibroblasts across non-failing (NF) and cardiomyopathy (CM: HCM + DCM) conditions using ParTIpy archetypal analysis. The workflow models fibroblast states as mixtures of **three** archetypal programs and quantifies disease-associated shifts in a continuous cell-state space. The code lives in `code/examples/fibroblast_cross_condition.py` and writes figures to `figures/fibroblast_cross_condition` and tabular outputs to `output/fibroblast_cross_condition`.

## Biological Context and Relevant Literature
This workflow emphasizes **continuous remodeling** of fibroblast programs rather than discrete cluster switching, aligning with recent cardiac sc/snRNA-seq studies.

Key biological points emphasized in the analysis:
- **Fibroblasts are central to cardiac remodeling.** They transition from quiescent ECM maintenance toward activated and stress-responsive programs in disease.
- **Cross-condition shifts are continuous.** Archetypal analysis captures gradual movement in low-dimensional state space.
- **Archetypal programs map to recognizable biology.** Activated fibrotic ECM programs (e.g., `POSTN`, `FGF14`, `FN1`, `COL1A1`, `VCAN`, `TNC`, `LOXL1`; TFs like `TWIST1`, `SCX`, `JUNB`, `HOXA5`); stress-responsive / regulatory programs (e.g., `FKBP5`, `PPARG`, `FOXO1`, `TGFBR2/3`, `SMAD3`, `SERPINE1`; TFs like `NR3C1`, `HIF1A`, `NFKB`, `CEBPB`, `MYC`, `NFE2L2`); perivascular / basement-membrane programs (e.g., `COL15A1`, `COLEC12`, `SMOC2`, `LAMB1`, `LAMC1`, `RGS5`; TFs like `RORA`, `NR2F2`, `PROX1`, `KLF17`).
- **TF and pathway enrichments anchor mechanism.** Collectri TF activity, PROGENy pathways, MSigDB Hallmarks, and Matrisome gene sets connect archetypes to signaling, metabolic, and ECM remodeling programs.

Relevant references (as cited in the manuscript draft):
- Chaffin M. et al. *Single-nucleus profiling of human dilated and hypertrophic cardiomyopathy.* Nature 608, 174–180 (2022).
- Patrick R. et al. *Integration mapping of cardiac fibroblast single-cell transcriptomes elucidates cellular principles of fibrosis in diverse pathologies.* Sci. Adv. 10, eadk8501 (2024).
- Lanzer J. D. et al. *A cross-study transcriptional patient map of heart failure defines conserved multicellular coordination in cardiac remodeling.* Nat. Commun. 16, 9659 (2025).
- Reichart D. et al. *Pathogenic variants damage cell composition and single cell transcription in cardiomyopathies.* Nature 2021. (https://www.nature.com/articles/s41586-021-03549-5)

## Inputs and Setup
- Input dataset: `data/human_dcm_hcm_scportal_03.17.2022.h5ad` (manual download noted in the script).
- Paths configured in `code/utils/const.py`: `DATA_PATH=data`, `FIGURE_PATH=figures`, `OUTPUT_PATH=output`.
- Cell types retained: `Fibroblast_I`, `Activated_fibroblast`, `Fibroblast_II`.
- Disease remapping: `HCM` → `CM`, `DCM` → `CM`, `NF` → `NF`.
- Color palette: `NF=#01665E`, `CM=#8C510A`.
- Archetype configuration: `n_archetypes=3`, `obsm_key=X_pca_harmony`, `obsm_dim=16`.
- Marker sets: generic cardiac markers for dotplot QC and fibroblast archetype marker blocks (3 blocks) with curated genes and TFs used for heatmaps.
- MSigDB is cached locally at `output/fibroblast_cross_condition/msigdb_raw.pkl` to avoid repeated downloads.

## Step-by-Step Analysis
1. **Load and subset fibroblasts.** Reads the `.h5ad` object and keeps fibroblast cell types. Validates that all generic marker genes exist in `adata.var_names`.
2. **Quality control filtering.** Filters cells with at least 100 detected genes and genes expressed in at least 10 cells.
3. **Normalization and feature selection.** Library-size normalizes, log1p transforms, computes highly variable genes, runs PCA (50 components) on HVGs, and z-scales expression (stored in `adata.layers["z_scaled"]`).
4. **Marker dotplot for QC.** Generates a dotplot of generic marker genes grouped by `donor_id` and saves it with Scanpy prefix `dotplot_`.
5. **Batch integration.** Runs Harmony on PCA coordinates using `donor_id` as the batch covariate, stores `X_pca_harmony`, and also computes `X_pca_harmony_tau1e3` for comparison.
6. **Choose AA dimensionality.** By default computes shuffled PCA (25 shuffles) and saves a plot, then uses the first 16 Harmony PCs for archetypal analysis. If `--quick` is passed, shuffled PCA is skipped.
7. **Select number of archetypes.** By default computes selection metrics for 2–7 archetypes and saves diagnostic plots; if `--quick` is passed, skips selection metrics and only runs bootstrap variance for `n_archetypes=3`.
8. **Archetype visualization.** Plots 2D archetype geometry with contours and 2D bootstrap stability, then generates disease-colored 2D projections on PC pairs (0,1), (0,2), (1,2).
9. **Meta-enrichment barplots.** Computes archetype weights, normalizes so each archetype’s weights sum to 1, and computes meta-enrichment for disease, original disease labels, cell types, and subclusters.
10. **Disease association testing (patient-level).** Aggregates to donor-level pseudobulk, projects onto the archetype convex hull, computes distances, produces point and box plots of distance vs disease, runs Welch’s t-test with BH correction, and fits ridge logistic models with permutation p-values and bootstrap confidence bands.
11. **Archetype gene expression characterization.** Computes archetype expression for `raw`, `log1p`, and `z_scaled`, writes a pivoted table, saves top genes per archetype (raw >= 1.0, top 50 by z-score), and generates a marker-gene heatmap.
12. **TF activity (Collectri, ULM).** Runs ULM to estimate TF activity per archetype, writes a TF activity table, saves top 15 significant TFs per archetype (`p<=0.05`), and plots a TF heatmap with significance marks (`p<=0.05`).
13. **Pathway activity (PROGENy, ULM).** Runs ULM to estimate pathway activity per archetype, clusters pathways for ordering, and plots a pathway heatmap with significance marks.
14. **Hallmark gene set activity (MSigDB Hallmark, ULM).** Filters hallmarks to sets with >=5 genes, runs ULM, writes the hallmark activity table, selects the top 5 positive hallmarks per archetype (`p<=0.05`), and plots a clustered heatmap.
15. **Matrisome gene set activity (MSigDB NABA, ULM).** Uses NABA Matrisome sets (excluding several cancer-related sets), runs ULM, writes the matrisome activity table, and plots a clustered heatmap.
16. **Save processed AnnData.** Writes processed AnnData to `output/fibroblast_cross_condition/fibroblast_cross_condition_partipy.h5ad` and to `/home/pschaefer/fibroblast_cross_condition_partipy.h5ad`.
17. **Integration vs no-integration sanity check.** Recomputes archetypes on unintegrated PCA and Harmony-integrated PCA, aligns archetypes by minimizing expression-distance, and saves correlation heatmaps and tables for `n_archetypes=3` and `n_archetypes=4`.

## Result Figures (Script Outputs)
These files are written to `figures/fibroblast_cross_condition` by the script.
- `figures/fibroblast_cross_condition/dotplot_marker_dotplot.pdf` — Dotplot of generic marker genes across donors.
- `figures/fibroblast_cross_condition/plot_shuffled_pca.pdf` — Shuffled PCA variance baseline for dimensionality choice (skipped with `--quick`).
- `figures/fibroblast_cross_condition/plot_var_explained.pdf` — Variance explained across archetype counts (2–7; not generated with `--quick`).
- `figures/fibroblast_cross_condition/plot_IC.pdf` — Information criterion across archetype counts (2–7; not generated with `--quick`).
- `figures/fibroblast_cross_condition/plot_bootstrap_variance.pdf` — Bootstrap variance across archetype counts (2–7; not generated with `--quick`).
- `figures/fibroblast_cross_condition/plot_archetypes_2D.pdf` — 2D archetype geometry (3 archetypes) with contours.
- `figures/fibroblast_cross_condition/plot_archetypes_2D.png` — PNG version of the 2D archetype geometry.
- `figures/fibroblast_cross_condition/plot_bootstrap_2D.pdf` — 2D bootstrap stability for archetypes.
- `figures/fibroblast_cross_condition/plot_bootstrap_2D.png` — PNG version of the bootstrap plot.
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_1.png` — Disease-colored 2D plot on PCs (0,1).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_2.pdf` — Disease-colored 2D plot on PCs (0,2).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_2.png` — PNG version.
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc1_pc_2.pdf` — Disease-colored 2D plot on PCs (1,2).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc1_pc_2.png` — PNG version.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_disease.pdf` — Archetype enrichment for CM vs NF.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_disease_original.pdf` — Archetype enrichment for original disease labels.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_celltypes_original.pdf` — Archetype enrichment by original cell type labels.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_cellstate_original.pdf` — Archetype enrichment by `SubCluster`.
- `figures/fibroblast_cross_condition/patient_pseudobulk_in_convex_hull.pdf` — Donor-level pseudobulk positions within the archetype convex hull.
- `figures/fibroblast_cross_condition/patient_pseudobulk_in_convex_hull_same_limits.pdf` — Same pseudobulk convex-hull plot using the PC0/PC1 axis limits from the single-cell disease plot.
- `figures/fibroblast_cross_condition/patient_pseudobulk_distance_point_plot.pdf` — Jittered distances to each archetype by disease.
- `figures/fibroblast_cross_condition/patient_pseudobulk_distance_boxplot.pdf` — Boxplots of distances to each archetype by disease.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_0.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 0.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_1.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 1.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_2.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 2.
- `figures/fibroblast_cross_condition/gene_expression_tile_plot.pdf` — Heatmap of archetype expression for curated fibroblast marker genes.
- `figures/fibroblast_cross_condition/tf_activation_tile_plot.pdf` — TF activity heatmap (Collectri ULM) with significance marks.
- `figures/fibroblast_cross_condition/progeny_tile_plot.pdf` — PROGENy pathway activity heatmap with significance marks.
- `figures/fibroblast_cross_condition/hallmark_tile_plot.pdf` — MSigDB Hallmark activity heatmap (top positive per archetype).
- `figures/fibroblast_cross_condition/matrisome_tile_plot.pdf` — Matrisome activity heatmap.
- `figures/fibroblast_cross_condition/aa_with_and_without_harmony_3.pdf` — Correlation heatmap comparing archetypes from PCA vs Harmony PCA (3 archetypes).
- `figures/fibroblast_cross_condition/aa_with_and_without_harmony_4.pdf` — Correlation heatmap comparing archetypes from PCA vs Harmony PCA (4 archetypes).

## Result Tables and Objects (Script Outputs)
These files are written to `output/fibroblast_cross_condition` by the script.
- `output/fibroblast_cross_condition/obs_aggregated.csv` — Donor-level pseudobulk metadata and distances to archetypes.
- `output/fibroblast_cross_condition/ttest_results.csv` — Welch’s t-test results for NF vs CM distances with BH correction.
- `output/fibroblast_cross_condition/disease_enrichment.csv` — Per-archetype ridge logistic regression summary. Columns: `archetype`, `term`, `coef_logit`, `OR`, `p_empirical`, `alpha_ridge`, `B_perm`, `n`, `q_empirical_BH`, `reject_q_0p05`.
- `output/fibroblast_cross_condition/archetype_expression_pivot.csv` — Archetype expression table (raw, log1p, z_scaled).
- `output/fibroblast_cross_condition/top_genes_archetype_arch_*.csv` — Top 50 genes per archetype (raw >= 1.0).
- `output/fibroblast_cross_condition/collectri_df.csv` — TF activity table from Collectri ULM. Columns: `archetype`, `TF`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/archetype_*_top15_TFs.csv` — Top 15 TFs per archetype among significant Collectri hits (`p<=0.05`), ranked by `t_value`.
- `output/fibroblast_cross_condition/progeny_df.csv` — PROGENy pathway activity table. Columns: `archetype`, `pathway`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/hallmark_df.csv` — MSigDB Hallmark activity table. Columns: `archetype`, `hallmark`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/matrisome_df.csv` — Matrisome activity table. Columns: `archetype`, `matrisome_set`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/aa_with_and_without_harmony_3.csv` — Correlation table for 3-archetype comparison (PCA vs Harmony).
- `output/fibroblast_cross_condition/aa_with_and_without_harmony_4.csv` — Correlation table for 4-archetype comparison (PCA vs Harmony).
- `output/fibroblast_cross_condition/msigdb_raw.pkl` — Cached MSigDB resource for repeated runs.
- `output/fibroblast_cross_condition/fibroblast_cross_condition_partipy.h5ad` — Processed AnnData object used for downstream plotting and tables.

## Notes and Potential Follow-Ups
- `--quick` skips shuffled PCA and archetype selection metrics, and only computes bootstrap variance for `n_archetypes=3` (without saving `plot_bootstrap_variance.pdf`).
- The script writes a copy of the processed AnnData to `/home/pschaefer/fibroblast_cross_condition_partipy.h5ad`, which is outside the repository.
