from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "bench_meta"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "bench_meta"
output_dir.mkdir(exist_ok=True, parents=True)

df_0 = pd.read_csv(Path("output") / "ms_bench" / "results.csv")
df_0["dataset"] = "ms"
df_0

df_1 = pd.read_csv(Path("output") / "ms_xenium_bench" / "results.csv")
df_1["dataset"] = "ms_xenium"
df_1

df_2 = pd.read_csv(Path("output") / "lupus_bench" / "results.csv")
df_2["dataset"] = "lupus"

df = pd.concat([
    df_0, df_1, df_2
])
df["setting"] = [ds + "__" + ct for ds, ct in zip(df["dataset"], df["celltype"])]

# compute mean and std for time and varexpl per setting 
mean_df = df.groupby("setting").aggregate({"varexpl": "mean", "time": "mean"})
mean_df.columns = [c + "_mean" for c in mean_df.columns]

std_df = df.groupby("setting").aggregate({"varexpl": "std", "time": "std"})
std_df.columns = [c + "_std" for c in std_df.columns]

df = df.join(mean_df, on="setting").join(std_df, on="setting")

df["varexpl_z_scaled"] = (df["varexpl"] - df["varexpl_mean"]) / df["varexpl_std"]
df["time_z_scaled"] = (df["time"] - df["time_mean"]) / df["time_std"]

df_summary = df.groupby(["setting", "init_alg", "optim_alg"]).aggregate({"varexpl_z_scaled": "mean", "time_z_scaled": "mean"}).reset_index()

df_summary_agg = df_summary.groupby(["init_alg", "optim_alg"]).aggregate({"varexpl_z_scaled": ["mean", "std"], "time_z_scaled": ["mean", "std"]}).reset_index()
df_summary_agg.columns = ['_'.join(col).strip('_') for col in df_summary_agg.columns.values]
df_summary_agg["varexpl_z_scaled_low"] = df_summary_agg["varexpl_z_scaled_mean"] - df_summary_agg["varexpl_z_scaled_std"]
df_summary_agg["varexpl_z_scaled_high"] = df_summary_agg["varexpl_z_scaled_mean"] + df_summary_agg["varexpl_z_scaled_std"]
df_summary_agg["time_z_scaled_low"] = df_summary_agg["time_z_scaled_mean"] - df_summary_agg["time_z_scaled_std"]
df_summary_agg["time_z_scaled_high"] = df_summary_agg["time_z_scaled_mean"] + df_summary_agg["time_z_scaled_std"]

color_map = {"frank_wolfe":  "#DAA520", "projected_gradients": "#006400"}

df_settings = df[["celltype", "dataset", "setting", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates().reset_index(drop=True)
df_settings["n_samples_log10"] = np.log10(df_settings["n_samples"])
df_settings["dataset_long"] = df_settings["dataset"].map({
    "ms": "10X Chromium White Matter\nMultiple Sclerosis",
    "ms_xenium": "10X Xenium Spinal Coord\nMultiple Sclerosis",
    "lupus": "10X Chromium PBMC\nSystemic Lupus Erythematosus"
})

p1 = (pn.ggplot(df_summary) 
     + pn.geom_hline(yintercept=0) 
     + pn.geom_vline(xintercept=0) 
     + pn.geom_point(data=df_summary, mapping=pn.aes(x="time_z_scaled", y="varexpl_z_scaled", color="optim_alg", shape="init_alg"), alpha=0.5)
     + pn.geom_point(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", y="varexpl_z_scaled_mean", color="optim_alg", shape="init_alg"), size=8)
     + pn.scale_color_manual(values=color_map)
     + pn.labs(x="Z-Scaled Time", y="Z-Scaled Variance Explained",
               color="Optimization Algorithm", shape="Initialization Algorithm")
     + pn.theme_bw()
     )
p1.save(figure_dir / "p1.pdf")

p2 = (pn.ggplot(df_summary) 
     + pn.geom_errorbar(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", ymin="varexpl_z_scaled_low", ymax="varexpl_z_scaled_high"), alpha=0.5, width=0)
     + pn.geom_errorbarh(data=df_summary_agg, mapping=pn.aes(y="varexpl_z_scaled_mean", xmin="time_z_scaled_low", xmax="time_z_scaled_high"), alpha=0.5, height=0)
     + pn.geom_point(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", y="varexpl_z_scaled_mean", color="optim_alg", shape="init_alg"), size=8)
     + pn.scale_color_manual(values=color_map)
     + pn.coord_equal()
     + pn.theme_bw()
     + pn.labs(x="Mean of Z-Scaled Time", y="Mean of Z-Scaled Variance Explained",
               color="Optimization Algorithm", shape="Initialization Algorithm")
     )
p2.save(figure_dir / "p2.pdf")

p3 = (pn.ggplot(df_settings) 
      + pn.geom_point(pn.aes(x="n_samples_log10", y="n_archetypes", color="dataset_long"), size=5, alpha=0.75) 
      + pn.labs(x="Number of Cells", y="Number of Archetypes", color="Dataset") 
      + pn.theme_bw() 
      + pn.theme(figure_size=(6.5, 4))
      )
p3.save(figure_dir / "p3.pdf")
