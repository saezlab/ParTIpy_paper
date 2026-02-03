# Fibroblast Cross-Condition Analysis Summary

## Overview
This analysis implements a cross-condition comparison of human cardiac fibroblasts using ParTIpy archetypal analysis. The workflow models fibroblast states as continuous mixtures of archetypal programs to quantify shifts between non-failing (NF) and cardiomyopathy (CM: HCM + DCM) conditions. The code runs in `code/examples/fibroblast_cross_condition.py` and writes figures to `figures/fibroblast_cross_condition` and tabular outputs to `output/fibroblast_cross_condition`.

## Biological Context and Relevant Literature
This workflow is designed to highlight **continuous disease-associated remodeling of fibroblast states** rather than discrete cluster shifts, which is well aligned with recent single-cell and single-nucleus cardiac studies.

Key biological points emphasized in the analysis:
- **Fibroblasts are central to cardiac remodeling.** They transition from quiescent ECM maintenance toward activated and myofibroblast-like programs in disease, producing fibrotic extracellular matrix, stress-response signatures, and pro-inflammatory signaling.
- **Cross-condition shifts are continuous.** Archetypal analysis captures gradual movement in a low-dimensional cell-state space, consistent with the continuum of fibroblast activation states reported in recent cardiac sc/snRNA-seq studies.
- **Archetypal programs map to recognizable biology.** The curated marker blocks in the script correspond to:
  - Quiescent/structural programs (e.g., `DCN`, `LUM`, `COL1A1/2`, `TCF21`, `NR2F2`).
  - Perivascular/vessel-associated programs (e.g., `COL15A1`, `COLEC12`, `RORA`, `HOXA5`).
  - Activated/myofibroblast ECM programs (e.g., `POSTN`, `THBS4`, `FAP`, `AEBP1`, `TWIST1`).
  - Stress/metabolic activation programs (e.g., `FKBP5`, `HIF3A`, `GPX3`, `NR3C1`, `HIF1A`).
- **TF and pathway enrichments give mechanistic anchors.** Collectri TF activity, PROGENy pathways, MSigDB Hallmarks, and Matrisome gene sets help connect archetypes to signaling, metabolic, and ECM remodeling programs that are frequently reported in cardiac fibrosis and heart failure literature.

Relevant references (as cited in the manuscript draft):
- Chaffin M. et al. *Single-nucleus profiling of human dilated and hypertrophic cardiomyopathy.* Nature 608, 174–180 (2022).  
  Source dataset for NF/HCM/DCM cardiac tissue used in this analysis.
- Patrick R. et al. *Integration mapping of cardiac fibroblast single-cell transcriptomes elucidates cellular principles of fibrosis in diverse pathologies.* Sci. Adv. 10, eadk8501 (2024).  
  Demonstrates fibroblast activation continua and conserved fibrotic programs across pathologies.
- Lanzer J. D. et al. *A cross-study transcriptional patient map of heart failure defines conserved multicellular coordination in cardiac remodeling.* Nat. Commun. 16, 9659 (2025).  
  Provides cross-study evidence for coordinated remodeling programs, motivating continuous-state analyses.

## Inputs and Setup
- Input dataset: `data/human_dcm_hcm_scportal_03.17.2022.h5ad` (manual download noted in the script).
- Paths configured in `code/utils/const.py`: `DATA_PATH=data`, `FIGURE_PATH=figures`, `OUTPUT_PATH=output`.
- Cell types retained: `Fibroblast_I`, `Activated_fibroblast`, `Fibroblast_II`.
- Disease remapping: `HCM` → `CM`, `DCM` → `CM`, `NF` → `NF`.
- Marker sets:
  - Generic cardiac markers for dotplot QC (fibroblast, endothelial, mural/pericyte, cardiomyocyte, immune).
  - Fibroblast archetype marker blocks (4 blocks) with curated genes and TFs used for later heatmaps.
- Color palette: `NF=#01665E`, `CM=#8C510A`.

## Step-by-Step Analysis
1. **Load and subset fibroblasts**
   - Reads the `.h5ad` object, filters to fibroblast cell types, and copies the subset.
   - Validates that all generic marker genes exist in `adata.var_names`.

2. **Quality control filtering**
   - Filters cells with at least 100 detected genes.
   - Filters genes expressed in at least 10 cells.

3. **Normalization and feature selection**
   - Library-size normalizes, log1p transforms, computes highly variable genes.
   - Runs PCA (50 components) on HVGs.
   - Z-scales expression (stored in `adata.layers["z_scaled"]`).

4. **Marker dotplot for QC**
   - Generates a dotplot of generic marker genes grouped by `donor_id`.
   - Output is saved by Scanpy with prefix `dotplot_`.

5. **Batch integration**
   - Runs Harmony on PCA coordinates using `donor_id` as the batch covariate.
   - Stores integrated PCs in `adata.obsm["X_pca_harmony"]`.

6. **Choose AA dimensionality**
   - Computes shuffled PCA (25 shuffles) to guide component selection.
   - Sets archetypal analysis input to the first 16 Harmony PCs.

7. **Select number of archetypes**
   - Computes selection metrics for 2–7 archetypes.
   - Saves variance explained, information criterion, and bootstrap variance plots.
   - Proceeds with **4 archetypes** for downstream analyses.

8. **Archetype visualization**
   - Plots 2D archetype geometry (with contours) for 4 archetypes.
   - Runs and visualizes bootstrap stability in 2D.
   - Generates disease-colored 2D projections on PC pairs (0,1), (0,2), (1,2).

9. **Meta-enrichment barplots**
   - Computes archetype weights and normalizes so each archetype’s weights sum to 1.
   - Computes meta-enrichment for disease, original disease labels, cell types, and subclusters.
   - Saves barplots for each enrichment view.

10. **Disease association testing (patient-level)**
    - Aggregates cell-level PCs to donor-level pseudobulk by averaging within `donor_id`.
    - Projects donor averages onto the archetype convex hull and computes distances to each archetype.
    - Fits ridge logistic regression per archetype distance predicting CM vs NF.
    - Uses 10,000 label permutations for empirical p-values and 1,000 bootstraps for a CI band over the fitted curve.
    - Saves per-archetype probability curves and writes a results table to CSV.

11. **Archetype gene expression characterization**
    - Computes mean z-scored expression per archetype for genes expressed in >100 cells.
    - Writes a long-form expression table (gene, archetype, value).
    - Generates a heatmap for curated fibroblast marker genes.

12. **TF activity (Collectri, ULM)**
    - Runs ULM to estimate TF activity per archetype.
    - Writes TF activity table with t-values and p-values.
    - Plots a TF heatmap for the curated TF list with significance marks (`p<=0.05`).

13. **Pathway activity (PROGENy, ULM)**
    - Runs ULM to estimate pathway activity per archetype.
    - Clusters pathways for a data-driven column ordering.
    - Writes pathway activity table and a pathway heatmap with significance marks.

14. **Hallmark gene set activity (MSigDB Hallmark, ULM)**
    - Filters hallmarks to sets with >=5 genes.
    - Runs ULM, writes hallmark activity table.
    - Selects top 5 positive hallmarks per archetype (`p<=0.05`), clusters hallmarks, and plots a heatmap with significance marks.

15. **Matrisome gene set activity (MSigDB NABA, ULM)**
    - Uses NABA Matrisome sets, excluding multiple cancer-related sets.
    - Runs ULM and writes the matrisome activity table.
    - Clusters matrisome sets and renders a heatmap with significance marks.

16. **Save processed AnnData**
    - Writes processed AnnData to `output/fibroblast_cross_condition/fibroblast_cross_condition_partipy.h5ad`.
    - Writes a second copy to `/home/pschaefer/fibroblast_cross_condition_partipy.h5ad`.

17. **Integration vs no-integration sanity check**
    - Recomputes archetypes on unintegrated PCA and on Harmony-integrated PCA.
    - Aligns archetypes by minimizing expression-distance, then computes expression correlations.
    - Saves an archetype correlation heatmap.

## Result Figures (Current Files)
These files are present in `figures/fibroblast_cross_condition`.
- `figures/fibroblast_cross_condition/dotplot_marker_dotplot.pdf` — Dotplot of generic marker genes across donors.
- `figures/fibroblast_cross_condition/plot_shuffled_pca.pdf` — Shuffled PCA variance baseline for dimensionality choice.
- `figures/fibroblast_cross_condition/plot_var_explained.pdf` — Variance explained across archetype counts (2–7).
- `figures/fibroblast_cross_condition/plot_IC.pdf` — Information criterion across archetype counts (2–7).
- `figures/fibroblast_cross_condition/plot_bootstrap_variance.pdf` — Bootstrap variance across archetype counts (2–7).
- `figures/fibroblast_cross_condition/plot_archetypes_2D.pdf` — 2D archetype geometry (4 archetypes) with contours.
- `figures/fibroblast_cross_condition/plot_archetypes_2D.png` — PNG version of the 2D archetype geometry.
- `figures/fibroblast_cross_condition/plot_bootstrap_2D.pdf` — 2D bootstrap stability for archetypes.
- `figures/fibroblast_cross_condition/plot_bootstrap_2D.png` — PNG version of the bootstrap plot.
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease.pdf` — Disease-colored 2D archetype plot (likely from a prior run or notebook).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_1.pdf` — Disease-colored 2D plot on PCs (0,1).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_1.png` — PNG version.
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_2.pdf` — Disease-colored 2D plot on PCs (0,2).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc0_pc_2.png` — PNG version.
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc1_pc_2.pdf` — Disease-colored 2D plot on PCs (1,2).
- `figures/fibroblast_cross_condition/plot_archetypes_2D_disease_pc1_pc_2.png` — PNG version.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_disease.pdf` — Archetype enrichment for CM vs NF.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_disease_original.pdf` — Archetype enrichment for original disease labels.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_celltypes_original.pdf` — Archetype enrichment by original cell type labels.
- `figures/fibroblast_cross_condition/barplot_meta_enrichment_cellstate_original.pdf` — Archetype enrichment by `SubCluster`.
- `figures/fibroblast_cross_condition/patient_pseudobulk_in_convex_hull.pdf` — Donor-level pseudobulk positions within the archetype convex hull.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_0.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 0.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_1.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 1.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_2.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 2.
- `figures/fibroblast_cross_condition/ridge_logit_bootCI_arch_3.pdf` — Ridge logistic fit for CM vs NF using distance to archetype 3.
- `figures/fibroblast_cross_condition/gene_expression_tile_plot.pdf` — Heatmap of archetype expression for curated fibroblast marker genes.
- `figures/fibroblast_cross_condition/tf_activation_tile_plot.pdf` — TF activity heatmap (Collectri ULM) with significance marks.
- `figures/fibroblast_cross_condition/progeny_tile_plot.pdf` — PROGENy pathway activity heatmap with significance marks.
- `figures/fibroblast_cross_condition/hallmark_tile_plot.pdf` — MSigDB Hallmark activity heatmap (top positive per archetype).
- `figures/fibroblast_cross_condition/matrisome_tile_plot.pdf` — Matrisome activity heatmap (file exists; script currently displays this plot rather than saving it).
- `figures/fibroblast_cross_condition/aa_with_and_without_harmony.pdf` — Correlation heatmap comparing archetypes from PCA vs Harmony PCA.

## Result Tables and Objects (Current Files)
These files are present in `output/fibroblast_cross_condition`.
- `output/fibroblast_cross_condition/disease_enrichment.csv` — Per-archetype ridge logistic regression summary.
  - Columns: `archetype`, `term`, `coef_logit`, `OR`, `p_empirical`, `alpha_ridge`, `B_perm`, `n`, `q_empirical_BH`, `reject_q_0p05`.
- `output/fibroblast_cross_condition/arch_expr_long.csv` — Long-form archetype expression table.
  - Columns: `gene`, `variable`, `value`.
  - Note: the script writes `archetype_expression.csv`; this file appears to be the same content under a different name.
- `output/fibroblast_cross_condition/collectri_df.csv` — TF activity table from Collectri ULM.
  - Columns: `archetype`, `TF`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/progeny_df.csv` — PROGENy pathway activity table.
  - Columns: `archetype`, `pathway`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/hallmark_df.csv` — MSigDB Hallmark activity table.
  - Columns: `archetype`, `hallmark`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/matrisome_df.csv` — Matrisome activity table.
  - Columns: `archetype`, `matrisome_set`, `t_value`, `p_value`.
- `output/fibroblast_cross_condition/fibroblast_cross_condition_partipy.h5ad` — Processed AnnData object used for downstream plotting and tables.

## Notes and Potential Follow-Ups
- The script writes a copy of the processed AnnData to `/home/pschaefer/fibroblast_cross_condition_partipy.h5ad`, which is outside the repository.
- The matrisome plot is generated with `p.show()` in the script; if you want consistent file output, update the script to explicitly save the plot (matching the existing `matrisome_tile_plot.pdf`).
- If you want to share a minimal package, the typical bundle is: all PDFs/PNGs in `figures/fibroblast_cross_condition`, the CSV tables in `output/fibroblast_cross_condition`, and the processed `.h5ad` file for reproducibility.
