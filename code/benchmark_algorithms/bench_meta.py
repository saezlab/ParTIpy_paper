# python -m code.benchmark_algorithms.bench_meta

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

df_1 = pd.read_csv(Path("output") / "ms_xenium_bench" / "results.csv")
df_1["dataset"] = "ms_xenium"

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

color_map = {"frank_wolfe":  "#DAA520", 
             "projected_gradients": "#006400"}

shape_map = {"furthest_sum": "o",
             "plus_plus": "^",
             "uniform": "s"
             }

shape_map_datasets = {
    "Lerma-Martin et al., 2024": "+",
    "Kukanja et al., 2024": "x",
    "Perez et al., 2022": "1",
}

df_settings = (df[["celltype", "dataset", "setting", "n_samples", "n_dimensions", "n_archetypes"]]
               .drop_duplicates()
               .reset_index(drop=True))
df_settings["n_samples_log10"] = np.log10(df_settings["n_samples"])

#df_settings["dataset_long"] = df_settings["dataset"].map({
#    "ms": "10X Chromium White Matter\nMultiple Sclerosis",
#    "ms_xenium": "10X Xenium Spinal Coord\nMultiple Sclerosis",
#    "lupus": "10X Chromium PBMC\nSystemic Lupus Erythematosus"
#})
df_settings["dataset_long"] = df_settings["dataset"].map({
    "ms": "Lerma-Martin et al., 2024",
    "ms_xenium": "Kukanja et al., 2024",
    "lupus": "Perez et al., 2022"
})

p1 = (pn.ggplot(df_summary) 
     + pn.geom_hline(yintercept=0) 
     + pn.geom_vline(xintercept=0) 
     + pn.geom_point(data=df_summary, mapping=pn.aes(x="time_z_scaled", y="varexpl_z_scaled", color="optim_alg", shape="init_alg"), alpha=0.5)
     + pn.geom_point(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", y="varexpl_z_scaled_mean", color="optim_alg", shape="init_alg"), size=8)
     + pn.scale_color_manual(values=color_map)
     + pn.scale_shape_manual(values=shape_map)
     + pn.labs(x="Z-Scaled Time", y="Z-Scaled Variance Explained",
               color="Optimization Algorithm", shape="Initialization Algorithm")
     + pn.theme_bw()
     )
p1.save(figure_dir / "p1.pdf")

p2 = (pn.ggplot(df_summary) 
     + pn.geom_errorbar(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", ymin="varexpl_z_scaled_low", ymax="varexpl_z_scaled_high"), alpha=0.5, width=0)
     + pn.geom_errorbarh(data=df_summary_agg, mapping=pn.aes(y="varexpl_z_scaled_mean", xmin="time_z_scaled_low", xmax="time_z_scaled_high"), alpha=0.5, height=0)
     + pn.geom_point(data=df_summary_agg, mapping=pn.aes(x="time_z_scaled_mean", y="varexpl_z_scaled_mean", color="optim_alg", shape="init_alg"), size=6)
     + pn.scale_color_manual(values=color_map)
     + pn.scale_shape_manual(values=shape_map)
     + pn.theme(figure_size=(6.5, 4))
     + pn.theme_bw()
     + pn.labs(x="Mean of z-scaled Time", y="Mean of z-scaled Variance Explained",
               color="Optimization Algorithm", shape="Initialization Algorithm")
     )
p2.save(figure_dir / "p2.pdf")

p3 = (pn.ggplot(df_settings) 
      + pn.geom_point(pn.aes(x="n_samples_log10", 
                             y="n_archetypes", 
                             shape="dataset_long"), 
                             size=5, 
                             alpha=1.0,
                             position=pn.position_jitter(height=0.10, random_state=1234)) 
      + pn.labs(x="Number of Cells", y="Number of Archetypes", shape="Dataset") 
      + pn.theme_bw() 
      + pn.theme(figure_size=(6.0, 4))
      + pn.scale_shape_manual(values=shape_map_datasets)
      )
p3.save(figure_dir / "p3.pdf")

# repeat the individual plots
for df_ds, name in zip((df_0, df_1, df_2), ("ms", "ms_xenium", "lupus")):
    df_ds["description"] = [celltype + " | " + str(n_samples) + " | " + str(n_arch) + " | " + str(n_dim) for celltype, n_samples, n_arch, n_dim in 
                                zip(df_ds["celltype"], df_ds["n_samples"], df_ds["n_archetypes"], df_ds["n_dimensions"])]
    df_ds["rss_norm"] = df_ds["rss"] / (df_ds["n_samples"] * df_ds["n_dimensions"])
    df_ds["key"] = [init + "__" + optim for init, optim in zip(df_ds["init_alg"], df_ds["optim_alg"])]

    # Grouping and aggregation
    settings = ["description", "init_alg", "optim_alg"]
    features = ["time", "rss", "varexpl", "rss_norm"]
    agg_df = df_ds.groupby(settings).agg({f: ["mean", "std"] for f in features})
    agg_df.columns = ['__'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index()

    # Precompute error bar limits
    agg_df["varexpl__ymin"] = agg_df["varexpl__mean"] - agg_df["varexpl__std"]
    agg_df["varexpl__ymax"] = agg_df["varexpl__mean"] + agg_df["varexpl__std"]
    agg_df["time__xmin"] = agg_df["time__mean"] - agg_df["time__std"]
    agg_df["time__xmax"] = agg_df["time__mean"] + agg_df["time__std"]

    p = (
        pn.ggplot(agg_df)
        + pn.geom_point(pn.aes(x="time__mean", y="varexpl__mean", color="optim_alg", shape="init_alg"), size=4)
        + pn.geom_errorbar(
            pn.aes(x="time__mean", ymin="varexpl__ymin", ymax="varexpl__ymax", color="optim_alg"), width=0)
        + pn.geom_errorbarh(
            pn.aes(y="varexpl__mean", xmin="time__xmin", xmax="time__xmax", color="optim_alg"), height=0)
        + pn.facet_wrap(facets="description", ncol=4, scales="free")
        + pn.labs(x="Time (s)", y="Variance Explained", 
                color="Optimization Algorithm", shape="Initialization Algorithm") 
        + pn.theme_bw() 
        + pn.theme(figure_size=(12, 6))
        + pn.scale_color_manual(values=color_map) 
        + pn.scale_shape_manual(values=shape_map)
        + pn.theme(legend_position="none")
    )
    p.save(figure_dir / f"{name}_individual_results.pdf")