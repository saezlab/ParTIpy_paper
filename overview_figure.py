from pathlib import Path

import partipy as pt
import scanpy as sc
import numpy as np
import plotnine as pn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
indices_inverse = np.array([idx for idx in range(N_SAMPLES) if idx not in indices])
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

pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=N_DIMENSION)

pt.compute_selection_metrics(adata, min_k=2, max_k=10)
pt.compute_bootstrap_variance(adata, n_bootstrap=20, n_archetypes_list=list(range(2, 11)))

pt.compute_archetypes(adata, n_archetypes=N_ARCHETYPES, verbose=True, archetypes_only=False)

p = (pt.plot_archetypes_2D(adata=adata)
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
     + pn.theme(figure_size=(6, 3), legend_position="none")
     )
p.save(figure_dir / "boostrap_variance.pdf")


pt.compute_archetype_weights(adata=adata, mode="automatic")
archetype_expression = pt.compute_archetype_expression(adata=adata)

arch_idx = 1

adata.obs[f"weights_archetype_{arch_idx}"] = adata.obsm["cell_weights"][:, arch_idx]
p = (pt.plot_archetypes_2D(adata=adata, color=f"weights_archetype_{arch_idx}")
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank())
     )
p.save(figure_dir / "archetype_2D_with_weight.pdf")

p = (pt.plot_archetypes_2D(adata=adata, color="condition", alpha=0.5) 
     + pn.theme_minimal() 
     + pn.theme(text=pn.element_blank()) 
     + pn.scale_color_manual(values=color_map)
     )
p.save(figure_dir / "archetype_2D_with_condition.pdf")

metaenrichment_df = pt.compute_meta_enrichment(adata=adata, meta_col="condition")
fig = pt.radarplot_meta_enrichment(metaenrichment_df, color_map=color_map)
fig.savefig(figure_dir / "radarplot_condition.pdf")

p = pt.barplot_meta_enrichment(metaenrichment_df, color_map=color_map)
p.save(figure_dir / "barplot_condition.pdf")

# generate random gene expression barplot
rng = np.random.default_rng(42)
n_genes = 8

df = pd.DataFrame({
    "Archetype 1 Gene Expression": rng.uniform(0, 1, n_genes) * 10
})

df = df.sort_values("Archetype 1 Gene Expression", ascending=False)
df["Gene"] = [f"Gene_{i}" for i in range(1, n_genes+1)]
df["Gene"] = pd.Categorical(df["Gene"], categories=df.sort_values("Archetype 1 Gene Expression")["Gene"], ordered=True)

p = (
    pn.ggplot(df)
    + pn.geom_col(pn.aes(y="Archetype 1 Gene Expression", x="Gene"), fill="grey", color="black", width=0.4)
    + pn.labs(y="Archetype 1 Gene Expression", x="")
    + pn.coord_flip()
    + pn.theme_minimal()
    + pn.theme(figure_size=(5, 3))
)
p.save(figure_dir / "gene_expression_barplot.pdf")

# generate random pathway enrichment barplot
rng = np.random.default_rng(42)
n_pathways = 4

df = pd.DataFrame({
    "Archetype 1 Pathway Enrichment Expression": rng.uniform(0, 1, n_genes) * 10
})

df = df.sort_values("Archetype 1 Pathway Enrichment Expression", ascending=False)
df["Pathway"] = [f"Pathway_{i}" for i in range(1, n_genes+1)]
df["Pathway"] = pd.Categorical(df["Pathway"], categories=df.sort_values("Archetype 1 Pathway Enrichment Expression")["Pathway"], ordered=True)

p = (
    pn.ggplot(df)
    + pn.geom_col(pn.aes(y="Archetype 1 Pathway Enrichment Expression", x="Pathway"), fill="grey", color="black", width=0.4)
    + pn.labs(y="Archetype 1 Pathway Enrichment Expression", x="")
    + pn.coord_flip()
    + pn.theme_minimal()
    + pn.theme(figure_size=(5, 3))
)
p.save(figure_dir / "pathway_enrichment_barplot.pdf")

# generate random archetype crosstalk graph
np.random.seed(42)

G = nx.DiGraph()
G.add_nodes_from([0, 1, 2])

possible_edges = [(i, j) for i in range(3) for j in range(3)]

num_edges = np.random.randint(1, len(possible_edges) + 1)
selected_edges = np.random.choice(len(possible_edges), size=num_edges, replace=False)

for idx in selected_edges:
    u, v = possible_edges[idx]
    weight = np.random.rand()
    G.add_edge(u, v, weight=weight)

weights = [G[u][v]['weight'] for u, v in G.edges()]

norm = mcolors.Normalize(vmin=min(weights), vmax=max(weights))
cmap = cm.viridis
edge_colors = [cmap(norm(G[u][v]['weight'])) for u, v in G.edges()]

pos =  nx.circular_layout(G)

fig, ax = plt.subplots(figsize=(10, 6))

node_degree = dict(G.degree(weight="weight"))
max_degree = max(node_degree.values()) if node_degree else 1
node_sizes = [800 + 200 * (node_degree[n] / max_degree) for n in G.nodes()]

nx.draw_networkx_nodes(
    G, pos, 
    node_color='lightblue', 
    node_size=node_sizes, 
    edgecolors='darkblue', 
    linewidths=2, 
    alpha=0.9, 
    ax=ax
)

nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', font_color='darkblue', ax=ax)

edge_widths = []
if weights:
    min_weight = min(weights)
    max_weight = max(weights)
    if max_weight == min_weight:
        edge_widths = [3.0] * len(weights)
    else:
        edge_widths = [2.0 + 4.0 * (w - min_weight) / (max_weight - min_weight) for w in weights]

nx.draw_networkx_edges(
    G,
    pos,
    edge_color=weights,
    edge_cmap=cmap,
    alpha=0.8,
    width=edge_widths,
    arrows=True,
    arrowsize=20,
    connectionstyle="arc3,rad=0.1",  # Small consistent curve for ALL edges
    min_source_margin=15,
    min_target_margin=15,
    ax=ax
)

# Colorbar
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#sm.set_array([])
#plt.colorbar(sm, ax=ax, label='Edge Weight', shrink=0.7)

plt.tight_layout()
fig.savefig(figure_dir / "crosstalk_graph.pdf")
