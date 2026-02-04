# manual download from here: https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy
# paper reference: https://www.nature.com/articles/s41586-022-04817-8
# other papers
# 1) https://www.nature.com/articles/s41586-021-03549-5
# 2)
# see Leo's analysis here: https://github.com/saezlab/best_practices_ParTIpy/tree/main
from pathlib import Path
import argparse

import plotnine as pn
import scanpy as sc
import partipy as pt

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from mizani.bounds import squish
import decoupler as dc
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
import harmonypy as hm
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind


from ..utils.const import FIGURE_PATH, OUTPUT_PATH, DATA_PATH

parser = argparse.ArgumentParser(
    description="Fibroblast cross-condition analysis workflow."
)
parser.add_argument(
    "--quick",
    action="store_true",
    help="Skip shuffled PCA computation and plotting.",
)
args = parser.parse_args()

########################################################################
# Setup paths
########################################################################
figure_dir = Path(FIGURE_PATH) / "fibroblast_cross_condition"
figure_dir.mkdir(exist_ok=True, parents=True)
sc.settings.figdir = figure_dir

output_dir = Path(OUTPUT_PATH) / "fibroblast_cross_condition"
output_dir.mkdir(exist_ok=True, parents=True)

########################################################################
# Setup color schemes
########################################################################
color_dict = {
    "NF": "#01665E",  # teal (blue-green)
    "CM": "#8C510A",  # brown
}

########################################################################
# General Configuration
########################################################################
n_archetypes = 3
obsm_key = "X_pca_harmony"
obsm_dim = 16

########################################################################
# Generic marker genes in cardio biology to check that data is clean
########################################################################
marker_dict_generic = {
    "Fibroblast": [
        "DCN",
        "LUM",
        "COL1A1",
        "COL1A2",
        "COL3A1",
        "PDGFRA",
        "TCF21",
    ],
    "Endothelial": [
        "PECAM1",
        "VWF",
        "KDR",
        "EMCN",
    ],
    "Mural / pericyte": [
        "RGS5",
        "PDGFRB",
        "CSPG4",
        "MCAM",
    ],
    "Cardiomyocyte": [
        "TNNT2",
        "MYH7",
        "ACTC1",
    ],
    "Immune": [
        "PTPRC",
        "LYZ",
    ],
}

########################################################################
# Fibroblast specific marker genes (already ordered by archetype)
########################################################################
marker_blocks = {
    2: {
        # Archetype 0
        "function": (
            "Activated fibrotic ECM fibroblasts / matrifibrocytes. "
            "Characterized by high fibrillar and provisional ECM production, "
            "active ECM remodeling and crosslinking, tissue stiffening, "
            "and engagement of mesenchymal activation programs."
        ),
        "genes": [
            # --- Core fibrotic / matrifibroblast ECM ---
            "POSTN",  # periostin; canonical activated fibroblast / matrifibrocyte marker
            # "FGF14",   # strong archetype-specific marker in this dataset; distinguishes this activated state
            "FN1",  # fibronectin; provisional ECM during wound repair
            "COL1A1",  # type I collagen; bulk fibrillar collagen deposition
            "COL12A1",  # FACIT collagen; regulates fibril organization and tissue mechanics
            "VCAN",  # versican; provisional ECM, cell–matrix interactions
            "TNC",  # tenascin C; injury-induced ECM, mechanotransduction
            "LOXL1",  # lysyl oxidase; collagen crosslinking and matrix stiffening
            # --- Pro-fibrotic signaling / matrix turnover ---
            # "CCN2",    # CTGF; downstream TGF-β effector, fibrosis amplification
            # "MRC2",    # collagen internalization and ECM turnover (boundary marker)
        ],
        "TFs": [
            # --- Mesenchymal activation / myofibroblast identity ---
            "TWIST1",  # EMT and mesenchymal reprogramming; fibroblast activation
            "SCX",  # tendon-like / ECM-specialized fibroblast lineage program
            # "JUNB",    # immediate-early AP-1 factor; generic activation and stress response
            # --- Chromatin permissiveness for activation ---
            "KAT6A",  # histone acetyltransferase; enables transcriptional activation
            "KDM2A",  # histone demethylase; chromatin remodeling during activation
            # --- Wound-associated mesenchymal programs ---
            # "HOXA5",   # positional fibroblast identity; often reactivated in fibrosis
            "PRRX1",  # wound-activated mesenchymal progenitor program
            "SRF",  # actin–cytoskeleton and mechanotransduction-driven transcription
        ],
    },
    0: {
        # Archetype 1
        "function": (
            "Stress-responsive / regulatory fibroblasts. "
            "Integrates hormonal, hypoxic, oxidative, and inflammatory stress signaling "
            "with intact TGF-β sensing, while remaining weakly fibrogenic."
        ),
        "genes": [
            # --- Hormonal and metabolic stress response ---
            "FKBP5",  # glucocorticoid receptor target; stress hormone responsiveness
            "PPARG",  # lipid metabolism and anti-fibrotic regulatory axis
            "FOXO1",  # stress tolerance, growth restraint, quiescence balance
            # --- TGF-β sensing machinery (buffered, not executed) ---
            "TGFBR2",  # TGF-β receptor II
            "TGFBR3",  # co-receptor; fine-tunes ligand availability and sensitivity
            "SMAD3",  # canonical TGF-β signal transducer
            # --- Canonical TGF-β output (weak / restrained) ---
            # "SERPINE1", # PAI-1; classic TGF-β–induced stress/fibrosis gene
            # --- Basement-membrane–associated collagen ---
            "COL4A4",  # type IV collagen; basement membrane, non-fibrillar ECM
        ],
        "TFs": [
            # --- Hormonal stress axis ---
            "NR3C1",  # glucocorticoid receptor; dominant stress-modulatory TF
            # --- Hypoxia and injury sensing ---
            "HIF1A",  # hypoxia-responsive transcription; ischemic and fibrotic stress
            # --- Inflammatory signal integration ---
            "NFKB",  # inflammatory master regulator
            "CEBPB",  # cytokine- and stress-responsive transcriptional amplifier
            # --- Metabolic and biosynthetic scaling ---
            "MYC",  # metabolic and transcriptional scaling under stress
            # --- Oxidative stress defense ---
            "NFE2L2",  # NRF2; redox balance and detoxification response
        ],
    },
    1: {
        # Archetype 2
        "function": (
            "Perivascular / basement-membrane fibroblasts. "
            "Vascular-adjacent niche fibroblasts characterized by laminin-rich "
            "and basement-membrane ECM, structural support, and limited fibrotic activation."
        ),
        "genes": [
            # --- Basement membrane / perivascular ECM ---
            "COL15A1",  # perivascular fibroblast hallmark collagen
            "COL4A1",  # type IV collagen; basement membrane scaffold
            "COLEC12",  # vascular-associated ECM receptor; endothelial adjacency
            "SMOC2",  # basement membrane–associated matricellular protein
            "LAMB1",  # laminin beta chain; basement membrane integrity
            "LAMC1",  # laminin gamma chain
            "ADAMTSL3",  # ECM-associated regulator; structural niche maintenance
            # --- Boundary marker ---
            # "RGS5",    # pericyte-adjacent marker (defines interface, not identity)
        ],
        "TFs": [
            # --- Positional and niche identity ---
            "PBX2",  # transcriptional cofactor; positional identity maintenance
            "LHX3",  # developmental positional program
            # --- Transcriptional restraint / chromatin control ---
            "HDAC3",  # chromatin repression; state stabilization
            "REST",  # global transcriptional silencing
            # --- Anti-activation / growth-factor buffering ---
            "KLF17",  # anti-EMT, maintains non-fibrotic state
            "ERF",  # MAPK-responsive transcriptional brake
            # --- Vascular / mesenchymal identity (optional) ---
            # "NR2F2",   # COUP-TFII; vascular-adjacent fibroblast identity
            # "PROX1",   # lymphatic / vascular niche transcriptional memory
        ],
    },
}


# ---------------------------------------------------------------------
# Derive ordered gene_list and tf_list for plotting (deduplicated)
# ---------------------------------------------------------------------
gene_list = []
for a in range(n_archetypes):
    gene_list += marker_blocks[a]["genes"]
seen = set()
gene_list = [g for g in gene_list if not (g in seen or seen.add(g))]

tf_list = []
for a in range(n_archetypes):
    tf_list += marker_blocks[a]["TFs"]
seen = set()
tf_list = [tf for tf in tf_list if not (tf in seen or seen.add(tf))]

########################################################################
# Plotting helper
########################################################################
marker_genes = []
var_group_positions = []
var_group_labels = []

start = 0
for group_label, genes in marker_dict_generic.items():
    marker_genes.extend(genes)
    end = start + len(genes) - 1
    var_group_positions.append((start, end))
    var_group_labels.append(group_label)
    start = end + 1

########################################################################
# Load PK
########################################################################
msigdb_cache_path = output_dir / "msigdb_raw.pkl"
needs_cache_write = False
if msigdb_cache_path.exists():
    msigdb_raw = pd.read_pickle(msigdb_cache_path)
else:
    msigdb_raw = dc.op.resource("MSigDB")
    needs_cache_write = True

if msigdb_raw.duplicated(["geneset", "genesymbol"]).any():
    msigdb_raw = msigdb_raw[~msigdb_raw.duplicated(["geneset", "genesymbol"])].copy()
    needs_cache_write = True

if needs_cache_write:
    msigdb_raw.to_pickle(msigdb_cache_path)

########################################################################
# Load and process the adata object
########################################################################
adata = sc.read_h5ad(Path(DATA_PATH) / "human_dcm_hcm_scportal_03.17.2022.h5ad")
print(adata.obs["cell_type_leiden0.6"].unique().to_list())
adata = adata[
    adata.obs["cell_type_leiden0.6"].isin(
        ["Fibroblast_I", "Activated_fibroblast", "Fibroblast_II"]
    ),
    :,
].copy()

missing = [g for g in marker_genes if g not in adata.var_names]
if missing:
    raise ValueError(f"Missing marker genes in adata.var_names: {missing}")

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
sc.pp.pca(adata, mask_var="highly_variable", n_comps=50)
adata.layers["z_scaled"] = sc.pp.scale(adata.X, max_value=10)

print(adata.obs["disease"].value_counts())

_ = sc.pl.dotplot(
    adata,
    var_names=marker_genes,
    groupby="donor_id",
    var_group_positions=var_group_positions,
    var_group_labels=var_group_labels,
    show=False,
    save="marker_dotplot.pdf",
)

# integration using harmony
ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs[["donor_id"]], "donor_id")
adata.obsm["X_pca_harmony"] = ho.Z_corr.copy()

ho2 = hm.run_harmony(adata.obsm["X_pca"], adata.obs[["donor_id"]], "donor_id", tau=1e3)
adata.obsm["X_pca_harmony_tau1e3"] = ho2.Z_corr.copy()
del ho, ho2

########################################################################
# Determine the number of principal components to consider for AA
########################################################################
if not args.quick:
    pt.compute_shuffled_pca(adata, mask_var="highly_variable", n_shuffle=25)
    p = pt.plot_shuffled_pca(adata) + pn.theme_bw()
    p.save(figure_dir / "plot_shuffled_pca.pdf", verbose=False)
else:
    print("Skipping shuffled PCA because --quick was set.")

pt.set_obsm(adata=adata, obsm_key=obsm_key, n_dimensions=obsm_dim)

########################################################################
# Determine the number of archetypes
########################################################################
if not args.quick:
    pt.compute_selection_metrics(adata=adata, n_archetypes_list=range(2, 8))

    p = pt.plot_var_explained(adata)
    p.save(figure_dir / "plot_var_explained.pdf", verbose=False)

    p = pt.plot_IC(adata)
    p.save(figure_dir / "plot_IC.pdf", verbose=False)

    pt.compute_bootstrap_variance(
        adata=adata, n_bootstrap=50, n_archetypes_list=range(2, 8)
    )

    p = pt.plot_bootstrap_variance(adata) + pn.theme_bw()
    p.save(figure_dir / "plot_bootstrap_variance.pdf", verbose=False)

else:
    pt.compute_bootstrap_variance(
        adata=adata, n_bootstrap=50, n_archetypes_list=[n_archetypes]
    )
    print("Skipping archetype selection metrics because --quick was set.")
p = (
    pt.plot_archetypes_2D(
        adata=adata,
        show_contours=True,
        result_filters={"n_archetypes": n_archetypes},
        alpha=0.05,
    )
    + pn.theme_bw()
)
p.save(figure_dir / "plot_archetypes_2D.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D.png", verbose=False)

p = (
    pt.plot_bootstrap_2D(adata, result_filters={"n_archetypes": n_archetypes})
    + pn.theme_bw()
)
p.save(figure_dir / "plot_bootstrap_2D.pdf", verbose=False)
p.save(figure_dir / "plot_bootstrap_2D.png", verbose=False)

########################################################################
# Some 2D plots
########################################################################
p = (
    pt.plot_archetypes_2D(
        adata=adata,
        dimensions=[0, 1],
        show_contours=True,
        color="disease",
        alpha=0.05,
        size=0.5,
        result_filters={"n_archetypes": n_archetypes},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
    + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1.0, "size": 5}))
    + pn.labs(x="PC 0", y="PC 1", color="Disease\nStatus")
    + pn.coord_equal()
    + pn.theme(
        legend_key=pn.element_rect(fill="white", color="white"),
        legend_background=pn.element_rect(fill="white", color="black"),
        axis_title_x=pn.element_text(size=16),
        axis_title_y=pn.element_text(size=16),
        axis_text=pn.element_text(size=13),
    )
)
p.save(figure_dir / "plot_archetypes_2D_disease_pc0_pc_1.png", verbose=False)

# save the limits for later
fig = p.draw()  # forces build + draw
ax = fig.axes[0]
pc_0_limits = ax.get_xlim()
pc_1_limits = ax.get_ylim()

p = (
    pt.plot_archetypes_2D(
        adata=adata,
        dimensions=[0, 2],
        show_contours=True,
        color="disease",
        alpha=0.05,
        size=0.5,
        result_filters={"n_archetypes": n_archetypes},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
    + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1.0, "size": 5}))
    + pn.labs(x="PC 0", y="PC 2", color="Disease\nStatus")
    + pn.coord_equal()
    + pn.theme(
        legend_key=pn.element_rect(fill="white", color="white"),
        legend_background=pn.element_rect(fill="white", color="black"),
    )
)
p.save(figure_dir / "plot_archetypes_2D_disease_pc0_pc_2.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D_disease_pc0_pc_2.png", verbose=False)

p = (
    pt.plot_archetypes_2D(
        adata=adata,
        dimensions=[2, 1],
        show_contours=True,
        color="disease",
        alpha=0.05,
        size=0.5,
        result_filters={"n_archetypes": n_archetypes},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
    + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1.0, "size": 5}))
    + pn.labs(x="PC 2", y="PC 1", color="Disease\nStatus")
    + pn.coord_equal()
    + pn.theme(
        legend_key=pn.element_rect(fill="white", color="white"),
        legend_background=pn.element_rect(fill="white", color="black"),
    )
)
p.save(figure_dir / "plot_archetypes_2D_disease_pc1_pc_2.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D_disease_pc1_pc_2.png", verbose=False)

########################################################################
# Some enrichment bar plots
########################################################################
pt.compute_archetype_weights(
    adata=adata, mode="automatic", result_filters={"n_archetypes": n_archetypes}
)
# NOTE: here we make sure that the weights per archetype sums to one
weights = pt.get_aa_cell_weights(adata, n_archetypes=n_archetypes)
weights /= weights.sum(axis=0, keepdims=True)
assert np.allclose(weights.sum(axis=0), 1, rtol=1e-3)

disease_enrichment = pt.compute_meta_enrichment(
    adata=adata, meta_col="disease", result_filters={"n_archetypes": n_archetypes}
)
p = pt.barplot_meta_enrichment(disease_enrichment, color_map=color_dict) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_disease.pdf", verbose=False)

disease_orginal_enrichment = pt.compute_meta_enrichment(
    adata=adata,
    meta_col="disease_original",
    result_filters={"n_archetypes": n_archetypes},
)
p = pt.barplot_meta_enrichment(disease_orginal_enrichment) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_disease_original.pdf", verbose=False)

ct_enrichment = pt.compute_meta_enrichment(
    adata=adata,
    meta_col="cell_type_leiden0.6",
    result_filters={"n_archetypes": n_archetypes},
)
p = pt.barplot_meta_enrichment(ct_enrichment) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_celltypes_original.pdf", verbose=False)

cs_enrichment = pt.compute_meta_enrichment(
    adata=adata, meta_col="SubCluster", result_filters={"n_archetypes": n_archetypes}
)
p = pt.barplot_meta_enrichment(cs_enrichment) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_cellstate_original.pdf", verbose=False)

########################################################################
# Assessing statistical significance of disease enrichment
########################################################################
# 0) setting
patient_col = "donor_id"
y_col = "disease"
y_col_one = "CM"
base_covars = ["sex"]

# guard: ensure covariates exist; coerce object covariates to categorical
for c in base_covars:
    if c not in adata.obs.columns:
        raise KeyError(
            f"Covariate '{c}' not found in adata.obs columns: {list(adata.obs.columns)}"
        )
    if str(adata.obs[c].dtype) == "object":
        adata.obs[c] = adata.obs[c].astype("category")

# 1) get the archetypes
aa_results_dict = pt.get_aa_result(adata, n_archetypes=n_archetypes)
Z = aa_results_dict["Z"].copy().astype(np.float32)
archetype_df = pd.DataFrame(
    Z, columns=[f"pc_{arch_idx}" for arch_idx in range(Z.shape[1])]
)
archetype_df["archetype"] = [f"{idx}" for idx in range(len(archetype_df))]
print(aa_results_dict.keys())
print(f"{Z.shape=}")

# 2) get the single-cell data (note adata.uns["AA_config"]["n_dimensions"] is a list of ints)
X = (
    adata.obsm[adata.uns["AA_config"]["obsm_key"]][
        :, adata.uns["AA_config"]["n_dimensions"]
    ]
    .copy()
    .astype(np.float32)
)
print(f"{X.shape=}")

# 3) aggregate per patient
cols_to_check = base_covars + [y_col] + [patient_col]
nuniq = adata.obs.groupby(patient_col)[cols_to_check].nunique(dropna=False)
assert (nuniq.max() <= 1).all(), (
    f"Non-unique covariates within sample_id:\n{nuniq.max()[nuniq.max() > 1]}"
)
obs_agg = adata.obs[cols_to_check].drop_duplicates().copy()
X_agg = np.zeros((len(obs_agg), X.shape[1]), dtype=np.float32)
for idx, smp_id in enumerate(obs_agg[patient_col]):
    X_agg[idx, :] = X[adata.obs[patient_col] == smp_id, :].mean(axis=0)
print(f"{X_agg.shape=}")

# 4) project onto the convex hull
AA_model = pt.AA(n_archetypes=len(Z))
AA_model.Z = Z
AA_model.n_samples = len(X_agg)
X_agg_proj = AA_model.transform(X_agg)
assert np.all(np.isclose(X_agg_proj.sum(axis=1), 1))
assert np.all(X_agg_proj >= 0)
print(f"{X_agg_proj.shape=}")
X_agg_conv = X_agg_proj @ Z
print(f"{X_agg_conv.shape=}")

# 5) add first 3 pcs to obs_agg for plotting
for pc_idx in [0, 1, 2]:
    obs_agg[f"pc_{pc_idx}"] = X_agg_conv[:, pc_idx]

# O) plotting
hull_df = None
if len(archetype_df) >= 3:
    hull = ConvexHull(archetype_df[["pc_0", "pc_1"]].to_numpy())
    hull_points = archetype_df.iloc[hull.vertices][["pc_0", "pc_1"]]
    hull_df = pd.concat([hull_points, hull_points.iloc[:1]], ignore_index=True)
p = (
    (
        pn.ggplot()
        + pn.geom_point(data=obs_agg, mapping=pn.aes(x="pc_0", y="pc_1", color=y_col))
        + pn.geom_point(
            data=archetype_df,
            mapping=pn.aes(x="pc_0", y="pc_1"),
            color="grey",
            alpha=0.5,
            size=2,
        )
        + pn.geom_label(
            data=archetype_df,
            mapping=pn.aes(x="pc_0", y="pc_1", label="archetype"),
            color="black",
            fill="white",
            boxcolor="black",
            label_r=0.2,
            label_padding=0.2,
            label_size=0.7,
            size=20,
        )
    )
    + pn.theme_bw()
    + pn.guides(color=pn.guide_legend(override_aes={"alpha": 1.0, "size": 5}))
    + pn.scale_color_manual(values=color_dict)
    + pn.labs(x="PC 0", y="PC 1", color="Disease\nStatus")
    + pn.theme(
        legend_key=pn.element_rect(fill="white", color="white"),
        legend_background=pn.element_rect(fill="white", color="black"),
    )
)
if hull_df is not None:
    p += pn.geom_path(
        data=hull_df,
        mapping=pn.aes(x="pc_0", y="pc_1"),
        color="grey",
        size=1.2,
        alpha=0.5,
    )
p.save(figure_dir / "patient_pseudobulk_in_convex_hull.pdf", verbose=False)

# now also save with the same limits as the single-cell scatter plot
p += pn.coord_cartesian(xlim=pc_0_limits, ylim=pc_1_limits)
p.save(figure_dir / "patient_pseudobulk_in_convex_hull_same_limits.pdf", verbose=False)


# 6) compute the distance matrix
dist_key = "euclidean"
D = cdist(XA=X_agg_conv, XB=Z, metric=dist_key)
print(f"{D.shape=}")

# 7) add distance to the dataframe
for arch_idx in range(len(Z)):
    obs_agg[f"dist_to_arch_{arch_idx}"] = D[:, arch_idx]

# O) More plotting
dist_cols = [c for c in obs_agg.columns if c.startswith("dist_to_arch_")]
plot_df = obs_agg.melt(id_vars=["donor_id", "disease"], value_vars=dist_cols)
plot_df["variable_clean"] = [
    s.replace("dist_to_arch_", "Archetype ") for s in plot_df["variable"]
]

p = (
    pn.ggplot(plot_df)
    + pn.geom_point(
        pn.aes(x="disease", y="value", color="disease"),
        show_legend=False,
        position=pn.position_jitter(width=0.10, height=0),
        size=2,
        alpha=0.5,
    )
    + pn.facet_wrap("variable_clean", ncol=n_archetypes)
    + pn.scale_color_manual(values=color_dict)
    + pn.theme_bw()
    + pn.theme(
        figure_size=(3, 3),
        strip_background=pn.element_rect(fill="white", color="black"),
        strip_text=pn.element_text(color="black"),
    )
    + pn.labs(x="Disease Status", y="Distance")
)
p.save(figure_dir / "patient_pseudobulk_distance_point_plot.pdf", verbose=False)


p = (
    pn.ggplot(plot_df)
    + pn.geom_boxplot(
        pn.aes(x="disease", y="value", color="disease"), show_legend=False
    )
    + pn.facet_wrap("variable_clean", ncol=n_archetypes)
    + pn.scale_color_manual(values=color_dict)
    + pn.theme_bw()
    + pn.theme(
        figure_size=(3, 3),
        strip_background=pn.element_rect(fill="white", color="black"),
        strip_text=pn.element_text(color="black"),
    )
    + pn.labs(x="Disease Status", y="Distance")
)
p.save(figure_dir / "patient_pseudobulk_distance_boxplot.pdf", verbose=False)

# 8) saving
obs_agg.to_csv(output_dir / "obs_aggregated.csv", index=False)

# 9) prepare covariates
cat_covars = [c for c in base_covars if str(obs_agg[c].dtype) in ("object", "category")]
num_covars = [c for c in base_covars if c not in cat_covars]


def covar_formula_terms(num_covars, cat_covars):
    terms = []
    terms += num_covars
    terms += [f"C({c})" for c in cat_covars]
    return terms


covar_terms = covar_formula_terms(num_covars, cat_covars)
dist_cols = [f"dist_to_arch_{k}" for k in range(len(Z))]
needed_cols = [y_col] + dist_cols + base_covars


# check for missing values
na_counts = obs_agg[needed_cols].isna().sum()
na_counts = na_counts[na_counts > 0]
if len(na_counts) > 0:
    raise ValueError(
        "Missing values detected in regression inputs:\n" + na_counts.to_string()
    )

# a) Welch's t-test
dist_cols = [c for c in obs_agg.columns if c.startswith("dist_to_arch_")]
g0, g1 = "NF", "CM"

# ---------------------------
rows = []
for c in dist_cols:
    x = obs_agg.loc[obs_agg["disease"] == g0, c].astype(float).dropna().to_numpy()
    y = obs_agg.loc[obs_agg["disease"] == g1, c].astype(float).dropna().to_numpy()

    tstat, pval = ttest_ind(x, y, equal_var=False, nan_policy="omit")  # Welch

    rows.append(
        {
            "dist_col": c,
            "group0": g0,
            "group1": g1,
            "n0": x.size,
            "n1": y.size,
            "mean0": np.mean(x) if x.size else np.nan,
            "mean1": np.mean(y) if y.size else np.nan,
            "diff_mean_0_minus_1": (np.mean(x) - np.mean(y))
            if (x.size and y.size)
            else np.nan,
            "t": tstat,
            "p": pval,
        }
    )

ttest_results = pd.DataFrame(rows)
reject, qvals, _, _ = multipletests(
    ttest_results["p"].values, alpha=0.05, method="fdr_bh"
)
ttest_results["q"] = qvals
ttest_results["reject_fdr_0p05"] = reject
ttest_results.to_csv(output_dir / "ttest_results.csv", index=False)

# O) boxplot with p-values from t-test
ann = (
    ttest_results.assign(
        variable_clean=lambda d: d["dist_col"].str.replace(
            "dist_to_arch_", "Archetype ", regex=False
        ),
        label=lambda d: d["p"].map(lambda p: f"p={p:.2g}"),
    )
    .loc[:, ["variable_clean", "label"]]
)
ypos = (
    plot_df.groupby("variable_clean")["value"]
    .max()
    .reset_index(name="ymax")
)
ypos["y"] = ypos["ymax"] * 1.03  # increase y-limit by ~8%

ann = ann.merge(ypos[["variable_clean", "y"]], on="variable_clean", how="left")

p = (
    pn.ggplot(plot_df)
    + pn.geom_boxplot(
        pn.aes(x="disease", y="value", color="disease"),
        show_legend=False,
    )
    + pn.geom_text(
        data=ann,
        mapping=pn.aes(x=1.5, y="y", label="label"),
        inherit_aes=False,
        size=8,
        ha="center",
        va="bottom",
    )
    + pn.facet_wrap("variable_clean", ncol=n_archetypes)
    + pn.scale_color_manual(values=color_dict)
    + pn.scale_y_continuous(expand=(0.05, 0.10))  # ← increases upper y-limit
    + pn.theme_bw()
    + pn.theme(
        figure_size=(3, 3),
        strip_background=pn.element_rect(fill="white", color="black"),
        strip_text=pn.element_text(color="black"),
    )
    + pn.labs(x="Disease Status", y="Distance")
)

p.save(
    figure_dir / "patient_pseudobulk_distance_boxplot_with_pvals.pdf",
    verbose=False,
)

# b) binary outcome regression
obs_agg["outcome"] = (obs_agg[y_col] == y_col_one).astype(int)
obs_agg["outcome"].value_counts()

alpha_ridge = 1e-3
B_perm = 10_000
B_boot = 1000
rng = np.random.default_rng(0)

# reference values for covariates (for prediction curve)
covar_ref_vals = {}
for c in num_covars:
    covar_ref_vals[c] = float(obs_agg[c].median())
for c in cat_covars:
    covar_ref_vals[c] = obs_agg[c].mode(dropna=False).iloc[0]


def fit_ridge_glm(data, formula, alpha):
    glm = smf.glm(formula=formula, data=data, family=sm.families.Binomial())
    return glm.fit_regularized(alpha=alpha, L1_wt=0.0)


def make_pred_df(dcol, d_grid, ref_vals):
    pred_df = pd.DataFrame({dcol: d_grid})
    for c, v in ref_vals.items():
        pred_df[c] = v
    return pred_df


arch_logit_results = []
plots = []

for k in range(len(Z)):
    dist_col = f"dist_to_arch_{k}"
    formula_rhs = " + ".join([dist_col] + (covar_terms if covar_terms else []))
    formula = f"outcome ~ {formula_rhs}"

    # --- observed fit ---
    fit_obs = fit_ridge_glm(obs_agg, formula, alpha_ridge)

    if dist_col not in fit_obs.params.index:
        raise KeyError(
            f"Term {dist_col} not found in fitted params. "
            f"Available terms: {list(fit_obs.params.index)}"
        )

    beta_hat = float(fit_obs.params[dist_col])

    # --- permutation null for beta (empirical p-value) ---
    y_orig = obs_agg["outcome"].to_numpy()
    beta_perm_dist = np.empty(B_perm, dtype=float)

    # reuse the same dataframe object; just overwrite the y column each time
    perm_df = obs_agg.copy()

    for b in range(B_perm):
        perm_df["outcome"] = rng.permutation(y_orig)
        fit_perm = fit_ridge_glm(perm_df, formula, alpha_ridge)
        beta_perm_dist[b] = float(fit_perm.params[dist_col])

    p_emp = (np.sum(np.abs(beta_perm_dist) >= np.abs(beta_hat)) + 1) / (B_perm + 1)

    arch_logit_results.append(
        {
            "archetype": k,
            "term": dist_col,
            "coef_logit": beta_hat,  # log(OR)
            "OR": float(np.exp(beta_hat)),
            "p_empirical": float(p_emp),
            "alpha_ridge": alpha_ridge,
            "B_perm": B_perm,
            "n": int(fit_obs.nobs) if hasattr(fit_obs, "nobs") else int(len(obs_agg)),
        }
    )

    # --- plot: observed probability curve + optional permutation envelope ---
    d_grid = np.linspace(obs_agg[dist_col].min(), obs_agg[dist_col].max(), 200)
    pred_df = make_pred_df(dist_col, d_grid, covar_ref_vals)

    p_hat = fit_obs.predict(pred_df)
    plot_df = pd.DataFrame({dist_col: d_grid, "p_hat": p_hat})

    # bootstrap CI band for p_hat(d)
    p_boot = np.empty((B_boot, len(d_grid)), dtype=float)
    idx_all = obs_agg.index.to_numpy()

    for b in range(B_boot):
        boot_idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        boot_df = obs_agg.loc[boot_idx].reset_index(drop=True)
        fit_boot = fit_ridge_glm(boot_df, formula, alpha_ridge)
        p_boot[b, :] = fit_boot.predict(pred_df)

    lo = np.quantile(p_boot, 0.025, axis=0)
    hi = np.quantile(p_boot, 0.975, axis=0)
    ci_df = pd.DataFrame({dist_col: d_grid, "lo": lo, "hi": hi})

    p_reg = (
        pn.ggplot()
        + pn.geom_jitter(
            data=obs_agg,
            mapping=pn.aes(x=dist_col, y="outcome"),
            height=0.0,
            alpha=0.25,
        )
        # bootstrap CI band
        + pn.geom_ribbon(
            data=ci_df, mapping=pn.aes(x=dist_col, ymin="lo", ymax="hi"), alpha=0.20
        )
        # fitted curve
        + pn.geom_line(data=plot_df, mapping=pn.aes(x=dist_col, y="p_hat"), size=1.2)
        + pn.labs(
            x=f"Distance to archetype {k}",
            y="P(CM)",
            title=(
                f"Ridge logistic + permutation test (arch {k})\n"
                f"bootstrap CI band for p_hat(d)\n"
                f"log(OR)={beta_hat:.3g}, OR={np.exp(beta_hat):.3g}, p_emp={p_emp:.3g}"
            ),
        )
        + pn.theme_bw()
    )

    # save + collect
    p_reg.save(figure_dir / f"ridge_logit_bootCI_arch_{k}.pdf", width=6, height=4)
    plots.append(p_reg)

uni_df = pd.DataFrame(arch_logit_results)

# optional: BH correction across archetypes
rej, qvals, _, _ = multipletests(uni_df["p_empirical"].values, method="fdr_bh")
uni_df["q_empirical_BH"] = qvals
uni_df["reject_q_0p05"] = rej

uni_df.to_csv(output_dir / "disease_enrichment.csv", index=False)

########################################################################
# Characterize archetypal gene expression
########################################################################
gene_mask = adata.var.index[adata.var["n_cells"] > 100].to_list()

arch_expr = {
    "raw": pt.compute_archetype_expression(adata=adata, layer="cellranger_raw")[
        gene_mask
    ],
    "log1p": pt.compute_archetype_expression(adata=adata, layer=None)[
        gene_mask
    ],  # assumes log1p in .X
    "z_scaled": pt.compute_archetype_expression(adata=adata, layer="z_scaled")[
        gene_mask
    ],
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

# Save top genes per archetype with a raw-expression threshold
raw_expr_threshold = 1.0
top_n_genes = 50
arch_expr_long_filtered = arch_expr_long.loc[
    arch_expr_long["raw"] >= raw_expr_threshold, :
].copy()
top_genes_df = (
    arch_expr_long_filtered.sort_values("z_scaled", ascending=False)
    .groupby("archetype", group_keys=False)
    .head(top_n_genes)
)
top_genes_df = top_genes_df.assign(
    rank=top_genes_df.groupby("archetype").cumcount() + 1
)
for arch, df_arch in top_genes_df.groupby("archetype"):
    df_arch.to_csv(output_dir / f"top_genes_archetype_{arch}_raw_ge_1.csv", index=False)

plot_df = arch_expr_long.loc[arch_expr_long["gene"].isin(gene_list), :].copy()
plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=gene_list, ordered=True)
plot_df["archetype"] = plot_df["archetype"].str.replace("arch_", "")
arch_order = [f"{idx}" for idx in range(n_archetypes)]
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)
p = (
    pn.ggplot(plot_df, pn.aes(y="gene", x="archetype", fill="z_scaled"))
    + pn.geom_tile()
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        limits=(-0.75, 0.75),
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        figure_size=(4, 9),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
    )
    + pn.labs(y="Gene", x="Archetype", fill="Mean z-scored\nGene Expression")
)
p.save(figure_dir / "gene_expression_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal TF activation
########################################################################
collectri = dc.op.collectri(organism="human")
collectri_acts_ulm_est, collectri_acts_ulm_est_p = dc.mt.ulm(
    data=arch_expr["z_scaled"], net=collectri, verbose=False
)

df_1 = collectri_acts_ulm_est.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="TF", value_name="t_value"
)
df_2 = collectri_acts_ulm_est_p.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="TF", value_name="p_value"
)
collectri_df = df_1.join(df_2.set_index(["archetype", "TF"]), on=["archetype", "TF"])
collectri_df.to_csv(output_dir / "collectri_df.csv", index=False)
del df_1, df_2

# save top 15 TFs per archetype
pval_thresh = 0.05
top_n = 15
df_filt = collectri_df[collectri_df["p_value"] <= pval_thresh].copy()

for arch, df_arch in df_filt.groupby("archetype"):
    top_tfs = df_arch.sort_values("t_value", ascending=False).head(top_n)
    top_tfs.to_csv(output_dir / f"archetype_{arch}_top{top_n}_TFs.csv", index=False)

plot_df = collectri_df
plot_df = plot_df.loc[plot_df["TF"].isin(tf_list), :].copy()
plot_df["TF"] = pd.Categorical(plot_df["TF"], categories=tf_list, ordered=True)
arch_order = [f"{idx}" for idx in range(n_archetypes)]
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)
p_sig = 0.05
sig_df = plot_df.loc[plot_df["p_value"] <= p_sig].copy()
p = (
    pn.ggplot(plot_df, pn.aes(y="TF", x="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.geom_text(
        data=sig_df,
        mapping=pn.aes(y="TF", x="archetype"),
        label="x",
        size=8,
        color="black",
    )
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        figure_size=(3, 9),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
    )
    + pn.labs(y="TF", x="Archetype", fill="TF Activation\nt-value")
)
p.save(figure_dir / "tf_activation_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal pathway activation (using progeny)
########################################################################
collectri = dc.op.progeny(organism="human")
progeny_acts_ulm_est, progeny_acts_ulm_est_p = dc.mt.ulm(
    data=arch_expr["z_scaled"], net=collectri, verbose=False
)

df_1 = progeny_acts_ulm_est.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="pathway", value_name="t_value"
)
df_2 = progeny_acts_ulm_est_p.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="pathway", value_name="p_value"
)
progeny_df = df_1.join(
    df_2.set_index(["archetype", "pathway"]), on=["archetype", "pathway"]
)
progeny_df.to_csv(output_dir / "progeny_df.csv", index=False)
del df_1, df_2

plot_df = progeny_df.copy()
arch_order = [f"{idx}" for idx in range(n_archetypes)]
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"].astype(str), categories=arch_order, ordered=True
)

wide = plot_df.pivot(index="archetype", columns="pathway", values="t_value")
wide = wide.loc[arch_order, :]

# Data-driven ordering by clustering progeny pathways (optimal leaf ordering)
col_linkage = sch.linkage(
    pdist(wide.T, metric="euclidean"),
    method="average",
    optimal_ordering=True,
)
pathway_order = wide.columns[sch.leaves_list(col_linkage)].tolist()
plot_df["pathway"] = pd.Categorical(
    plot_df["pathway"], categories=pathway_order, ordered=True
)

sig_df = plot_df.loc[plot_df["p_value"] <= p_sig].copy()

p = (
    pn.ggplot(plot_df, pn.aes(y="pathway", x="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.geom_text(
        data=sig_df,
        mapping=pn.aes(y="pathway", x="archetype"),
        label="x",
        size=8,
        color="black",
    )
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        figure_size=(6, 9),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
    )
    + pn.labs(y="Pathway", x="Archetype", fill="Progeny Enrichment\nt-value")
)
p.save(figure_dir / "progeny_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal functions using MSigDB hallmarks
########################################################################
min_genes_per_pathway = 5
max_genes_per_pathway = np.inf

database = "hallmark"
msigdb = msigdb_raw[msigdb_raw["collection"] == database]
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

hallmark_acts_ulm_est, hallmark_acts_ulm_est_p = dc.mt.ulm(
    data=arch_expr["z_scaled"], net=msigdb, verbose=False
)
df_t = hallmark_acts_ulm_est.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="hallmark", value_name="t_value"
)
df_p = hallmark_acts_ulm_est_p.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="hallmark", value_name="p_value"
)
hallmark_df = df_t.merge(df_p, on=["archetype", "hallmark"], how="inner")
hallmark_df.to_csv(output_dir / "hallmark_df.csv", index=False)
del df_t, df_p

hallmark_df["archetype"] = hallmark_df["archetype"].astype(str)

# Selection that guarantees hallmarks for every archetype
n_per_arch = 5
mode = "pos"
p_cutoff = 0.05

sel_df = hallmark_df.copy()

if p_cutoff is not None:
    sel_df = sel_df.loc[sel_df["p_value"] <= p_cutoff, :].copy()

if mode == "pos":
    sel_df = sel_df.loc[sel_df["t_value"] > 0, :].copy()
    sel_df["score"] = sel_df["t_value"]
elif mode == "abs":
    sel_df["score"] = sel_df["t_value"].abs()
else:
    raise ValueError("mode must be one of {'pos', 'abs'}")

top_per_arch = (
    sel_df.sort_values("score", ascending=False)
    .groupby("archetype", group_keys=False)
    .head(n_per_arch)
)

top_hallmarks = top_per_arch["hallmark"].unique().tolist()

plot_df = hallmark_df.loc[hallmark_df["hallmark"].isin(top_hallmarks), :].copy()
wide = plot_df.pivot(index="archetype", columns="hallmark", values="t_value")
wide = wide.loc[[str(i) for i in range(n_archetypes)], :]

# Data-driven ordering by clustering hallmarks (optimal leaf ordering)
col_linkage = sch.linkage(
    pdist(wide.T, metric="euclidean"),
    method="average",
    optimal_ordering=True,
)
hallmark_order = wide.columns[sch.leaves_list(col_linkage)].tolist()

arch_order = [str(i) for i in range(n_archetypes)]

label_map = {h: h.replace("HALLMARK_", "") for h in hallmark_order}
plot_df["hallmark"] = pd.Categorical(
    plot_df["hallmark"], categories=hallmark_order, ordered=True
)
plot_df["hallmark_label"] = plot_df["hallmark"].map(label_map)
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)
sig_df = plot_df.loc[plot_df["p_value"] <= p_sig].copy()

p = (
    pn.ggplot(plot_df, pn.aes(y="hallmark_label", x="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.geom_text(
        data=sig_df,
        mapping=pn.aes(y="hallmark_label", x="archetype"),
        label="x",
        size=8,
        color="black",
    )
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        figure_size=(6, 9),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
    )
    + pn.labs(y="Hallmark", x="Archetype", fill="Hallmark Enrichment\nt-value")
)
p.save(figure_dir / "hallmark_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal functions using MSigDB NABA_MATRISOME
########################################################################
naba_cancer_sets = [
    "NABA_MATRISOME_PRIMARY_METASTATIC_COLORECTAL_TUMOR",
    "NABA_MATRISOME_HIGHLY_METASTATIC_BREAST_CANCER",
    "NABA_MATRISOME_HIGHLY_METASTATIC_BREAST_CANCER_TUMOR_CELL_DERIVED",
    "NABA_MATRISOME_POORLY_METASTATIC_BREAST_CANCER",
    "NABA_MATRISOME_POORLY_METASTATIC_BREAST_CANCER_TUMOR_CELL_DERIVED",
    "NABA_MATRISOME_HIGHLY_METASTATIC_MELANOMA",
    "NABA_MATRISOME_HIGHLY_METASTATIC_MELANOMA_TUMOR_CELL_DERIVED",
    "NABA_MATRISOME_POORLY_METASTATIC_MELANOMA",
    "NABA_MATRISOME_POORLY_METASTATIC_MELANOMA_TUMOR_CELL_DERIVED",
    "NABA_MATRISOME_METASTATIC_COLORECTAL_LIVER_METASTASIS",
    "NABA_MATRISOME_HGSOC_OMENTAL_METASTASIS",
    "NABA_MATRISOME_MULTIPLE_MYELOMA",
]

matrisome = msigdb_raw.loc[msigdb_raw["geneset"].str.startswith("NABA_"), :].copy()
matrisome = matrisome.loc[~matrisome["geneset"].isin(naba_cancer_sets), :].copy()
matrisome = matrisome.rename(columns={"geneset": "source", "genesymbol": "target"})
matrisome_acts_ulm_est, matrisome_acts_ulm_est_p = dc.mt.ulm(
    data=arch_expr["z_scaled"],
    net=matrisome,
    verbose=False,
)

df_t = matrisome_acts_ulm_est.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="matrisome_set", value_name="t_value"
)
df_p = matrisome_acts_ulm_est_p.reset_index(names="archetype").melt(
    id_vars="archetype", var_name="matrisome_set", value_name="p_value"
)

matrisome_df = df_t.merge(df_p, on=["archetype", "matrisome_set"], how="inner")

matrisome_df.to_csv(output_dir / "matrisome_df.csv", index=False)
del df_t, df_p

plot_df = matrisome_df.copy()
arch_order = [f"{idx}" for idx in range(n_archetypes)]
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)

wide = plot_df.pivot(index="archetype", columns="matrisome_set", values="t_value")
wide = wide.loc[arch_order, :]

# Data-driven ordering by clustering matrisome gene sets (optimal leaf ordering)
col_linkage = sch.linkage(
    pdist(wide.T, metric="euclidean"),
    method="average",
    optimal_ordering=True,
)
matrisome_order = wide.columns[sch.leaves_list(col_linkage)].tolist()
matrisome_order = [s.replace("NABA_", "") for s in matrisome_order]
plot_df["matrisome_set"] = plot_df["matrisome_set"].str.replace("NABA_", "")
plot_df["matrisome_set"] = pd.Categorical(
    plot_df["matrisome_set"], categories=matrisome_order, ordered=True
)

p_sig = 0.05
sig_df = plot_df.loc[plot_df["p_value"] <= p_sig].copy()

p = (
    pn.ggplot(plot_df, pn.aes(y="matrisome_set", x="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.geom_text(
        data=sig_df,
        mapping=pn.aes(y="matrisome_set", x="archetype"),
        label="x",
        size=8,
        color="black",
    )
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        figure_size=(6, 9),
        axis_title_x=pn.element_text(size=14),
        axis_title_y=pn.element_text(size=14),
        axis_text_x=pn.element_text(size=11),
        axis_text_y=pn.element_text(size=11),
        legend_title=pn.element_text(size=12),
        legend_text=pn.element_text(size=9),
    )
    + pn.labs(y="Gene", x="Archetype", fill="Matrisome Terms\nt-value")
)
p.save(figure_dir / "matrisome_tile_plot.pdf", verbose=False)

########################################################################
# Save processed adata object (for now both on sds and locally)
########################################################################
pt.write_h5ad(adata, output_dir / "fibroblast_cross_condition_partipy.h5ad")
pt.write_h5ad(adata, "/home/pschaefer/fibroblast_cross_condition_partipy.h5ad")


########################################################################
# Integration vs No-Integration Test
########################################################################
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


gene_mask = adata.var.index[adata.var["n_cells"] > 100].to_list()

keys_to_delete = [
    "AA_results",
    "AA_config",
    "AA_bootstrap",
    "AA_cell_weights",
    "AA_selection_metrics",
]
for n_archetypes_test in [3, 4]:
    adata_default = adata.copy()
    for k in keys_to_delete:
        if k in adata_default.uns:
            del adata_default.uns[k]
    pt.set_obsm(adata=adata_default, obsm_key="X_pca", n_dimensions=16)
    pt.compute_archetypes(adata_default, n_archetypes=n_archetypes_test)
    pt.compute_archetype_weights(
        adata_default,
        result_filters={"n_archetypes": n_archetypes_test, "obsm_key": "X_pca"},
    )
    weights = pt.get_aa_cell_weights(
        adata_default, n_archetypes=n_archetypes_test, obsm_key="X_pca"
    )
    weights /= weights.sum(axis=0, keepdims=True)
    assert np.allclose(weights.sum(axis=0), 1, rtol=1e-3)
    archetype_expression_default = pt.compute_archetype_expression(
        adata=adata_default,
        layer="z_scaled",
        result_filters={"n_archetypes": n_archetypes_test},
    )[gene_mask]

    adata_harmony = adata.copy()
    for k in keys_to_delete:
        if k in adata_harmony.uns:
            del adata_harmony.uns[k]
    pt.set_obsm(adata=adata_harmony, obsm_key="X_pca_harmony", n_dimensions=16)
    pt.compute_archetypes(adata_harmony, n_archetypes=n_archetypes_test)
    pt.compute_archetype_weights(
        adata_harmony,
        result_filters={"n_archetypes": n_archetypes_test, "obsm_key": "X_pca_harmony"},
    )
    weights = pt.get_aa_cell_weights(
        adata_harmony, n_archetypes=n_archetypes_test, obsm_key="X_pca_harmony"
    )
    weights /= weights.sum(axis=0, keepdims=True)
    assert np.allclose(weights.sum(axis=0), 1, rtol=1e-3)
    archetype_expression_harmony = pt.compute_archetype_expression(
        adata=adata_harmony,
        layer="z_scaled",
        result_filters={"n_archetypes": n_archetypes_test},
    )[gene_mask]

    dist = cdist(
        archetype_expression_default, archetype_expression_harmony, metric="correlation"
    )
    corr = 1 - dist
    _ref_idx, query_idx = linear_sum_assignment(dist)

    archetype_expression_aligned_a = (
        archetype_expression_default.loc[_ref_idx, :].to_numpy().copy()
    )
    archetype_expression_aligned_b = (
        archetype_expression_harmony.loc[query_idx, :].to_numpy().copy()
    )

    pearson_r, pearson_pvals = pearsonr_per_row(
        archetype_expression_aligned_a, archetype_expression_aligned_b, return_pval=True
    )

    plot_df = (
        pd.DataFrame(
            corr[:, query_idx],
            index=[f"{idx}" for idx in range(n_archetypes_test)],
            columns=[f"{idx}" for idx in range(n_archetypes_test)],
        )
        .reset_index(names="x")
        .melt(id_vars="x", var_name="y", value_name="correlation")
    )
    plot_df.to_csv(output_dir / f"aa_with_and_without_harmony_{n_archetypes_test}.csv")
    p = (
        pn.ggplot(plot_df)
        + pn.geom_tile(pn.aes(x="x", y="y", fill="correlation"))
        + pn.geom_text(
            pn.aes(x="x", y="y", label="correlation"), format_string="{:.2f}"
        )
        + pn.scale_fill_gradient2(
            low="#2166AC",
            mid="#FFFFFF",
            high="#B2182B",
            midpoint=0,
            limits=(-1.0, 1.0),
            oob=squish,
        )
        + pn.labs(
            x="Archetypes based on X_pca",
            y="Archetypes based on X_pca_harmony",
            fill="Gene Expression\nCorrelation",
        )
    )
    p.save(figure_dir / f"aa_with_and_without_harmony_{n_archetypes_test}.pdf")
    del plot_df
