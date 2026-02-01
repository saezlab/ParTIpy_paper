# manual download from here: https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy
# paper reference: https://www.nature.com/articles/s41586-022-04817-8
# see Leo's analysis here: https://github.com/saezlab/best_practices_ParTIpy/tree/main
from pathlib import Path

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


from ..utils.const import FIGURE_PATH, OUTPUT_PATH, DATA_PATH

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
# Generic marker genes in cardio biology
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
    0: {
        "function": "quiescent / signaling; axon-guidance & ion-channel enriched",
        "genes": [
            "NRP1",
            "CNTNAP2",
            "SLIT3",
            "KCND2",
            "KCND3",
            "CACNA2D3",
            "ANK3",
            "FBLN1",
            # "LSAMP",
            # "NAV3",
            # "FLRT2",
            # "GRID2",
            # "ANK2",
            # "PDZRN4",
            # "SMAD6",
            # "BICC1",
        ],
        "TFs": [
            "NR2F2",  # vascular / mesenchymal lineage
            "EMX2",  # developmental patterning
            "SP1",  # basal transcription / promoter occupancy
            "LMO2",  # lineage regulation
            "BRCA1",  # genome integrity / quiescence
            "PROX1",  # fibroblast / vascular identity
            "TP53",  # stress surveillance
            "SRF",  # structural maintenance (low-activation context)
        ],
    },
    1: {
        "function": "perivascular / vessel-associated fibroblasts",
        "genes": [
            "COL15A1",
            "COLEC12",
            "FREM1",
            "PDE5A",
            "KCNMB2",
            "KCNN3",
            "SOX6",
            "TCF4",
        ],
        "TFs": [
            "RORA",  # niche / circadian modulation
            "LHX3",  # developmental patterning
            "ATOH1",  # differentiation control
            "MKX",  # fibroblast identity restraint
            "ETV6",  # vascular-associated regulation
            "KLF17",  # anti-activation TF
            "NEUROD2",  # positional identity reuse
            "HOXA5",  # anterior–posterior patterning
        ],
    },
    2: {
        "function": "activated fibroblast → myofibroblast / matrifibrocyte axis (fibrotic ECM)",
        "genes": [
            "POSTN",
            "THBS4",
            "FAP",
            "COL22A1",
            "AEBP1",
            "ITGA10",
            "LTBP2",
            "WISP2",
        ],
        "TFs": [
            "MYF5",  # deep mesenchymal program
            "TWIST1",  # EMT / activation
            "BHLHA15",  # differentiation switch
            "SCX",  # myofibroblast / tendon-like program
            "SRF",  # contractility (high-activation context)
            "KDM2A",  # chromatin remodeling
            "KAT6A",  # histone acetylation
            "JUNB",  # immediate-early activation
        ],
    },
    3: {
        "function": "stress-responsive / metabolic activation (redox, hypoxia, detox)",
        "genes": [
            "FKBP5",
            "HIF3A",
            "TXNRD1",
            "GPX3",
            "NNMT",
            "MGST1",
            "CFD",
            "GLUL",
            # "ADH1B",
        ],
        "TFs": [
            "NR3C1",  # glucocorticoid stress response (strongest overall)
            "HIF1A",  # hypoxia
            "PPARD",  # lipid metabolism
            "NFKB",  # inflammation
            "CEBPB",  # inflammatory transcription
            "MYC",  # metabolic scaling
            "NFE2L2",  # oxidative stress (NRF2)
            "FOXO3",  # stress resilience
        ],
    },
}
gene_list = []
for a in [0, 1, 2, 3]:
    gene_list += marker_blocks[a]["genes"]
# de-duplicate while preserving order
seen = set()
gene_list = [g for g in gene_list if not (g in seen or seen.add(g))]

tf_list = [tf for a in [0, 1, 2, 3] for tf in marker_blocks[a]["TFs"]]
seen = set()
tf_list = [tf for tf in tf_list if not (tf in seen or seen.add(tf))]

########################################################################
# Setup prior knowledge marker TFs and pathways
########################################################################
fibro_tf_dict = {
    "quiescent-like": [
        "NR3C1",
        "PPARA",
        "PPARD",
        "FOXO3",
    ]
}

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
sc.pp.pca(adata, mask_var="highly_variable")
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

########################################################################
# Determine the number of principal components to consider for AA
########################################################################
pt.compute_shuffled_pca(adata, mask_var="highly_variable", n_shuffle=25)
p = pt.plot_shuffled_pca(adata) + pn.theme_bw()
p.save(figure_dir / "plot_shuffled_pca.pdf", verbose=False)

pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=16)

########################################################################
# Determine the number of archetypes
########################################################################
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

p = (
    pt.plot_archetypes_2D(
        adata=adata, show_contours=True, result_filters={"n_archetypes": 4}, alpha=0.05
    )
    + pn.theme_bw()
)
p.save(figure_dir / "plot_archetypes_2D.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D.png", verbose=False)

p = pt.plot_bootstrap_2D(adata, result_filters={"n_archetypes": 4}) + pn.theme_bw()
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
        result_filters={"n_archetypes": 4},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
)
p.save(figure_dir / "plot_archetypes_2D_disease_pc0_pc_1.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D_disease_pc0_pc_1.png", verbose=False)

p = (
    pt.plot_archetypes_2D(
        adata=adata,
        dimensions=[0, 2],
        show_contours=True,
        color="disease",
        alpha=0.05,
        size=0.5,
        result_filters={"n_archetypes": 4},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
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
        result_filters={"n_archetypes": 4},
    )
    + pn.theme_bw()
    + pn.scale_color_manual(values=color_dict)
)
p.save(figure_dir / "plot_archetypes_2D_disease_pc1_pc_2.pdf", verbose=False)
p.save(figure_dir / "plot_archetypes_2D_disease_pc1_pc_2.png", verbose=False)

########################################################################
# Some enrichment bar plots
########################################################################
pt.compute_archetype_weights(
    adata=adata, mode="automatic", result_filters={"n_archetypes": 4}
)
# NOTE: here we make sure that the weights per archetype sums to one
weights = pt.get_aa_cell_weights(adata, n_archetypes=4)
weights /= weights.sum(axis=0, keepdims=True)
assert np.allclose(weights.sum(axis=0), 1, rtol=1e-3)

disease_enrichment = pt.compute_meta_enrichment(
    adata=adata, meta_col="disease", result_filters={"n_archetypes": 4}
)
p = pt.barplot_meta_enrichment(disease_enrichment, color_map=color_dict) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_disease.pdf", verbose=False)

disease_orginal_enrichment = pt.compute_meta_enrichment(
    adata=adata, meta_col="disease_original", result_filters={"n_archetypes": 4}
)
p = pt.barplot_meta_enrichment(disease_orginal_enrichment) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_disease_original.pdf", verbose=False)

ct_enrichment = pt.compute_meta_enrichment(
    adata=adata, meta_col="cell_type_leiden0.6", result_filters={"n_archetypes": 4}
)
p = pt.barplot_meta_enrichment(ct_enrichment) + pn.theme_bw()
p.save(figure_dir / "barplot_meta_enrichment_celltypes_original.pdf", verbose=False)

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
aa_results_dict = pt.get_aa_result(adata, n_archetypes=4)
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
    + pn.scale_color_manual(values=color_dict)
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

# 6) compute the distance matrix
dist_key = "euclidean"
D = cdist(XA=X_agg_conv, XB=Z, metric=dist_key)
print(f"{D.shape=}")

# 7) add distance to the dataframe
for arch_idx in range(len(Z)):
    obs_agg[f"dist_to_arch_{arch_idx}"] = D[:, arch_idx]

# 8) prepare covariates
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

# binary outcome regression
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
            height=0.05,
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
archetype_expression = pt.compute_archetype_expression(adata=adata, layer="z_scaled")[
    gene_mask
]

arch_expr_long = archetype_expression.copy().T
arch_expr_long.columns = [f"arch_{c}" for c in arch_expr_long.columns]
arch_expr_long = arch_expr_long.reset_index(names="gene")
arch_expr_long = arch_expr_long.melt(id_vars="gene")
arch_expr_long.to_csv(output_dir / "arch_expr_long.csv", index=False)

plot_df = arch_expr_long.loc[arch_expr_long["gene"].isin(gene_list), :].copy()
plot_df["gene"] = pd.Categorical(plot_df["gene"], categories=gene_list, ordered=True)
arch_order = [f"arch_{idx}" for idx in range(4)]
plot_df["variable"] = pd.Categorical(
    plot_df["variable"], categories=arch_order, ordered=True
)
p = (
    pn.ggplot(plot_df, pn.aes(x="gene", y="variable", fill="value"))
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
        axis_text_x=pn.element_text(rotation=90, ha="center"),
        figure_size=(9, 3),
    )
    + pn.labs(x="Gene", y="Archetype", fill="Mean z-scored\nGene Expression")
)
p.save(figure_dir / "gene_expression_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal TF activation
########################################################################
collectri = dc.op.collectri(organism="human")
collectri_acts_ulm_est, collectri_acts_ulm_est_p = dc.mt.ulm(
    data=archetype_expression, net=collectri, verbose=False
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

plot_df = collectri_df
plot_df = plot_df.loc[plot_df["TF"].isin(tf_list), :].copy()
plot_df["TF"] = pd.Categorical(plot_df["TF"], categories=tf_list, ordered=True)
arch_order = [f"{idx}" for idx in range(4)]
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)
p = (
    pn.ggplot(plot_df, pn.aes(x="TF", y="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        axis_text_x=pn.element_text(rotation=90, ha="center"),
        figure_size=(9, 3),
    )
    + pn.labs(x="Gene", y="Archetype", fill="TF Activation\nt-value")
)
p.save(figure_dir / "tf_activation_tile_plot.pdf", verbose=False)

########################################################################
# Characterize archetypal pathway activation (using progeny)
########################################################################
collectri = dc.op.progeny(organism="human")
progeny_acts_ulm_est, progeny_acts_ulm_est_p = dc.mt.ulm(
    data=archetype_expression, net=collectri, verbose=False
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

########################################################################
# Characterize archetypal functions using MSigDB hallmarks
########################################################################
database = "hallmark"
min_genes_per_pathway = 5
max_genes_per_pathway = np.inf
msigdb_raw = dc.op.resource("MSigDB")
msigdb_raw = msigdb_raw[~msigdb_raw.duplicated(["geneset", "genesymbol"])].copy()

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
    data=archetype_expression, net=msigdb, verbose=False
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

wide = wide.loc[[str(i) for i in range(4)], :]

# Data-driven ordering by clustering hallmarks (optimal leaf ordering)
col_linkage = sch.linkage(
    pdist(wide.T, metric="euclidean"),
    method="average",
    optimal_ordering=True,
)
hallmark_order = wide.columns[sch.leaves_list(col_linkage)].tolist()

arch_order = [str(i) for i in range(4)]

label_map = {h: h.replace("HALLMARK_", "") for h in hallmark_order}
plot_df["hallmark"] = pd.Categorical(
    plot_df["hallmark"], categories=hallmark_order, ordered=True
)
plot_df["hallmark_label"] = plot_df["hallmark"].map(label_map)
plot_df["archetype"] = pd.Categorical(
    plot_df["archetype"], categories=arch_order, ordered=True
)

p = (
    pn.ggplot(plot_df, pn.aes(x="hallmark_label", y="archetype", fill="t_value"))
    + pn.geom_tile()
    + pn.scale_fill_gradient2(
        low="#2166AC",
        mid="#FFFFFF",
        high="#B2182B",
        midpoint=0,
        oob=squish,
    )
    + pn.theme_bw()
    + pn.theme(
        axis_text_x=pn.element_text(rotation=90, ha="center"),
        figure_size=(9, 6),
    )
    + pn.labs(x="Hallmark", y="Archetype", fill="Hallmark Enrichment\nt-value")
)

p.show()

########################################################################
# Save processed adata object (for now both on sds and locally)
########################################################################
pt.write_h5ad(adata, output_dir / "fibroblast_cross_condition_partipy.h5ad")
pt.write_h5ad(adata, "/home/pschaefer/fibroblast_cross_condition_partipy.h5ad")
