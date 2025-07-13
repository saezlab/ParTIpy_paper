from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.pyplot as plt
from statsmodels.gam.api import GLMGam, BSplines
from statsmodels.genmod.families import Gaussian

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "coreset_meta"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "coreset_meta"
output_dir.mkdir(exist_ok=True, parents=True)

df_0_plotting = pd.read_csv(Path("output") / "ms_coreset" / "results.csv")
df_0_meta = pd.read_csv(Path("output") / "ms_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_0 = pd.read_csv(Path("output") / "ms_coreset" / "time_savings.csv")
assert df_0.shape[0] == df_0_meta.shape[0]
df_0["dataset"] = "ms"
df_0 = df_0.join(df_0_meta.set_index("celltype"), on="celltype", how="left")

df_1_plotting = pd.read_csv(Path("output") / "lupus_coreset" / "results.csv")
df_1_meta = pd.read_csv(Path("output") / "lupus_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_1 = pd.read_csv(Path("output") / "lupus_coreset" / "time_savings.csv")
assert df_1.shape[0] == df_1_meta.shape[0]
df_1["dataset"] = "lupus"
df_1 = df_1.join(df_1_meta.set_index("celltype"), on="celltype", how="left")

df_2_plotting = pd.read_csv(Path("output") / "ms_xenium_coreset" / "results.csv")
df_2_meta = pd.read_csv(Path("output") / "ms_xenium_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_2 = pd.read_csv(Path("output") / "ms_xenium_coreset" / "time_savings.csv")
assert df_2.shape[0] == df_2_meta.shape[0]
df_2["dataset"] = "ms_xenium"
df_2 = df_2.join(df_2_meta.set_index("celltype"), on="celltype", how="left")

df = pd.concat([df_0, df_1, df_2])
df["log10_n_samples"] = np.log10(df["n_samples"])

df.to_csv(output_dir / "summary.csv")

archetype_colors = {
    3: "#DBEAFE",  # Light blue (more blue than very light)
    4: "#93C5FD",  # Medium-light blue
    5: "#60A5FA",  # Medium blue
    6: "#2563EB",  # Blue
    7: "#1D4ED8",  # Dark blue
    8: "#1E3A8A"   # Very dark blue
}

df["n_archetypes"] = pd.Categorical(df["n_archetypes"])

p = (pn.ggplot(df, mapping=pn.aes(x="n_samples", y="time_saving")) 
     + pn.geom_point(pn.aes(color="n_archetypes"), size=5) 
     + pn.geom_smooth(method="lm") 
     + pn.scale_x_log10()
     + pn.labs(x="Number of Cells", y="Coreset Time Saving", color="Number of\nArchetypes")
     + pn.scale_color_manual(values=archetype_colors)
     + pn.theme_bw()
     + pn.theme(axis_text=pn.element_text(size=12), axis_title=pn.element_text(size=16))
     )
p.save(figure_dir / "time_saving_vs_number_of_cells.pdf", dpi=300, verbose=True)

p = (pn.ggplot(df, mapping=pn.aes(x="n_samples", y="coreset_size")) 
     + pn.geom_point(pn.aes(color="n_archetypes"), size=5) 
     + pn.geom_smooth(method="lm") 
     + pn.scale_x_log10()
     + pn.scale_y_log10()
     + pn.labs(x="Number of Cells", y="Minimal Required Coreset Fraction", color="Number of\nArchetypes")
     + pn.scale_color_manual(values=archetype_colors)
     + pn.theme_bw()
     + pn.theme(axis_text=pn.element_text(size=12), axis_title=pn.element_text(size=16))
     )
p.save(figure_dir / "minimal_coreset_fraction_vs_number_of_cells.pdf", dpi=300, verbose=True)

ncols = 4

for df_ds, name in zip((df_0_plotting, df_1_plotting, df_2_plotting), 
                       ("ms", "ms_xenium", "lupus")):

     n_celltypes = df_ds["celltype"].nunique()
     nrows = int(np.ceil(n_celltypes / ncols))

     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5*nrows))
     axes = axes.flatten()

     min_coresets_list = []

     for (celltype, df_group), ax in zip(df_ds.groupby("celltype"), axes):

          df_group = df_group.copy()
          
          n_samples = df_group["n_samples"].to_numpy()[0]
          n_dimensions = df_group["n_dimensions"].to_numpy()[0]
          n_archetypes = df_group["n_archetypes"].to_numpy()[0]

          df_group["log10_coreset_fraction"] = np.log10(df_group["coreset_fraction"])
          top_varexpl = df_group.loc[df_group["coreset_fraction"] == 1.0]["varexpl"].mean()
          deviation_fraction = 0.01
          target_varexpl = top_varexpl - (top_varexpl * deviation_fraction)
          df_group = df_group[["log10_coreset_fraction", "varexpl"]].dropna()
          df_group = df_group.sort_values("log10_coreset_fraction")

          # extract the variables
          x = df_group["log10_coreset_fraction"].values
          y = df_group["varexpl"].values

          # Create B-spline basis functions
          # You can adjust the number of knots and degree as needed
          n_splines = 7 # Number of basis functions
          spline_basis = BSplines(x, df=n_splines, degree=2)

          # Add intercept term
          intercept = np.ones((len(x), 1))

          # Fit the GAM model with intercept
          gam_model = GLMGam(y, exog=intercept, smoother=spline_basis, family=Gaussian())
          gam_results = gam_model.fit()

          # Generate predictions for plotting
          x_pred = np.linspace(x.min(), x.max(), 100)

          # Create intercept for predictions
          intercept_pred = np.ones((len(x_pred), 1))

          # Get predictions using the model"s predict method
          y_pred_mean = gam_results.predict(exog=intercept_pred, exog_smooth=x_pred.reshape(-1, 1))

          # For confidence intervals, we"ll use a simpler approach
          # Calculate standard errors manually or use bootstrap if needed
          # For now, let"s create a basic confidence interval using residual std
          residual_std = np.std(gam_results.resid_response)
          y_pred_ci = 1.96 * residual_std  # Approximate 95% CI

          # Residual analysis
          residuals = gam_results.resid_response
          fitted_values = gam_results.fittedvalues

          # some more things
          min_corset_size = x_pred[y_pred_mean > target_varexpl].min()
          min_varexpl = y_pred_mean[y_pred_mean > target_varexpl].min()

          min_coresets_list.append({
               "celltype": celltype,
               "log10_coreset_size": min_corset_size,
               "coreset_size": 10**min_corset_size,
               "varexpl": min_varexpl,
          })

          # Plot the results
          ax.vlines(x=0, 
                    ymin=top_varexpl-(top_varexpl * deviation_fraction), 
                    ymax=top_varexpl+(top_varexpl * deviation_fraction),
                    color="black")
          ax.scatter(x, y, alpha=0.5, label="Data points")
          ax.plot(x_pred, y_pred_mean, "r-", linewidth=2, label="GAM fit")
          ax.fill_between(x_pred, y_pred_mean - y_pred_ci, y_pred_mean + y_pred_ci, 
                              alpha=0.3, color="red", label="Approx 95% CI")
          ax.axvline(min_corset_size, color="green")
          ax.axhline(min_varexpl, color="green")
          ax.set_title(f"{celltype} | {n_samples} | min corset fraction: {10**min_corset_size:.2f}")
          ax.legend()
          ax.grid(True, alpha=0.3)

     fig.savefig(figure_dir / f"{name}_varexpl_vs_coreset_fraction_gam.pdf")
     plt.close()
