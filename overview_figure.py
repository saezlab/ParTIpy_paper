from pathlib import Path

import partipy as pt
import scanpy as sc
import numpy as np
import plotnine as pn
import pandas as pd
import matplotlib.pyplot as plt

from const import FIGURE_PATH

figure_dir = Path(FIGURE_PATH) / "overview_figure"
figure_dir.mkdir(exist_ok=True, parents=True)

N_SAMPLES = 1_000
N_DIMENSION = 4
N_ARCHETYPES = 3
NOISE_STD = 0.15
SEED = 153

X, A, _ = pt.simulate_archetypes(n_samples=N_SAMPLES, n_archetypes=N_ARCHETYPES, n_dimensions=N_DIMENSION, noise_std=NOISE_STD, seed=SEED)
X = X.astype(np.float32)

adata = sc.AnnData(X=X.copy())

rng = np.random.default_rng(42)
p = A[:, 0].copy()
p /= sum(p)
indices = rng.choice(N_SAMPLES, size=N_SAMPLES//2, replace=False, p=p)
indices_inverse = np.array([idx for idx in range(N_SAMPLES) if not idx in indices])
adata.obs["condition"] = ["healthy" if idx in indices else "disease" for idx in range(N_SAMPLES)]

color_map = {
    "healthy": "#1b9e77",  # Teal green
    "disease":  "#d95f02",  # Warm orange
}

sc.pp.pca(adata, n_comps=N_DIMENSION, svd_solver="auto")

X = adata.obsm["X_pca"].copy()
data_df = pd.DataFrame(X[:, :2], columns=["x0", "x1"])
p = (pn.ggplot(data_df) 
     + pn.geom_point(mapping=pn.aes(x="x0", y="x1"), color="black", alpha=0.5) 
     + pn.coord_equal() 
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank()))
p.save(figure_dir / "pca_2D.pdf")

pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimension=N_DIMENSION)

pt.compute_selection_metrics(adata)
pt.compute_bootstrap_variance(adata, n_bootstrap=20)

pt.compute_archetypes(adata, n_archetypes=N_ARCHETYPES, verbose=True, archetypes_only=False)

p = (pt.plot_archetypes_2D(adata=adata, show_two_panels=False) 
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank())
     )
p.save(figure_dir / "archetype_2D.pdf")

p = (pt.plot_var_explained(adata) 
     + pn.theme_minimal() 
     + pn.theme(figure_size=(6, 3))
     )
p.save(figure_dir / "var_explained.pdf")

p = (pt.plot_IC(adata) 
     + pn.theme_minimal() 
     + pn.theme(figure_size=(6, 3))
     )
p.save(figure_dir / "IC.pdf")

p = (pt.plot_bootstrap_variance(adata) 
     + pn.theme_minimal() 
     + pn.theme(figure_size=(6, 3))
     )


pt.compute_archetype_weights(adata=adata, mode="automatic")
archetype_expression = pt.compute_archetype_expression(adata=adata)

arch_idx = 1

adata.obs[f"weights_archetype_{arch_idx}"] = adata.obsm["cell_weights"][:, arch_idx]
p = (pt.plot_archetypes_2D(adata=adata, color=f"weights_archetype_{arch_idx}", show_two_panels=False) 
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank())
     )
p.save(figure_dir / "archetype_2D_with_weight.pdf")

p = (pt.plot_archetypes_2D(adata=adata, show_two_panels=False, color="condition", alpha=0.5) 
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank()) 
     + pn.scale_color_manual(values=color_map)
     )
p.save(figure_dir / "archetype_2D_with_condition.pdf")

metaenrichment_df = pt.compute_meta_enrichment(adata=adata, meta_col="condition")
fig = pt.radarplot_meta_enrichment(metaenrichment_df, color_map=color_map)
fig.savefig(figure_dir / "radarplot_condition.pdf")
