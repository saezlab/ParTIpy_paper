import numpy as np
import plotnine as pn
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

from partipy.paretoti import var_explained_aa

def plot_projected_dist(
    adata: sc.AnnData,
) -> pn.ggplot:
    """
    Create a plot showing the projected distance for a range of archetypes.
    The data is retrieved from `adata.uns["AA_var"]`. If `AA_var` is not found, `var_explained_aa` is called.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData object containing the variance explained data in `adata.uns["AA_var"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the projected distance plot.
    """
    # Validation input
    if "AA_var" not in adata.uns:
        print("AA_var not found in adata.uns. Computing variance explained by archetypal analysis...")
        var_explained_aa(adata=adata)

    plot_df = adata.uns["AA_var"]

    p = (
        pn.ggplot(plot_df)
        + pn.geom_col(mapping=pn.aes(x="k", y="dist_to_projected"))
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.labs(x="Number of Archetypes (k)", y="Distance to Projected Point")
        + pn.theme_matplotlib()
    )

    return p


def plot_var_on_top(
    adata: sc.AnnData,
) -> pn.ggplot:
    """
    Generate a plot showing the additional variance explained by AA models when increasing the number
    of archetypes from `k-1` to `k`    The data is retrieved from `adata.uns["AA_var"]`. If `AA_var` is not found, `var_explained_aa` is called.

    Parameters
    ----------
    adata : sc.AnnData
        AnnData objectt containing the variance explained data in `adata.uns["AA_var"]`.

    Returns
    -------
    pn.ggplot
        A ggplot object showing the variance explained on top of (k-1) model plot.
    """
    # Validation input
    if "AA_var" not in adata.uns:
        print("AA_var not found in adata.uns. Computing variance explained by archetypal analysis...")
        var_explained_aa(adata=adata)

    plot_df = adata.uns["AA_var"]

    p = (
        pn.ggplot(plot_df)
        + pn.geom_point(pn.aes(x="k", y="varexpl_ontop"), color="black")
        + pn.geom_line(pn.aes(x="k", y="varexpl_ontop"), color="black")
        + pn.labs(x="Number of Archetypes (k)", y="Variance Explained on Top of (k-1) Model")
        + pn.scale_x_continuous(breaks=list(np.arange(plot_df["k"].min(), plot_df["k"].max() + 1)))
        + pn.lims(y=(0, None))
        + pn.theme_matplotlib()
    )

    return p

def radarplot_meta_enrichment(meta_enrich: pd.DataFrame):
    """
    Parameters
    ----------
    meta_enrich: pd.DataFrame
        Output of meta_enrichment(), a pd.DataFrame containing the enrichment of meta categories (columns) for all archetypes (rows).

    Returns
    -------
    plt.pyplot.Figure
        Radar plots for all archetypes.
    """
    # Prepare data
    meta_enrich = meta_enrich.T.reset_index().rename(columns={"index": "Meta_feature"})

    # Function to create a radar plot for a given row
    def make_radar(row, title, color):
        # Set number of meta categories
        categories = list(meta_enrich)[1:]
        N = len(categories)

        # Calculate angles for the radar plot
        angles = [n / float(N) * 2 *  np.pi for n in range(N)]
        angles += angles[:1]

        # Initialise the radar plot
        ax = plt.subplot(int(np.ceil(len(meta_enrich) / 2)), 2, row + 1, polar=True)

        # Put first axis on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # One axe per variable and add labels
        archetype_label = [f"A{i}" for i in range(len(list(meta_enrich)[1:]))]
        plt.xticks(angles[:-1], archetype_label, color="grey", size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks(
            [0, 0.25, 0.5, 0.75, 1],
            ["0", "0.25", "0.50", "0.75", "1.0"],
            color="grey",
            size=7,
        )
        plt.ylim(0, 1)

        # Draw plot
        values = meta_enrich.loc[row].drop("Meta_feature").values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        plt.title(title, size=11, color=color, y=1.065)

    # Initialize the figure
    my_dpi = 96
    plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

    # Create a color palette:
    my_palette = plt.colormaps.get_cmap("Dark2")

    # Loop to plot
    for row in range(0, len(meta_enrich.index)):
        make_radar(
            row=row,
            title=f"Feature: {meta_enrich['Meta_feature'][row]}",
            color=my_palette(row),
        )

    return plt