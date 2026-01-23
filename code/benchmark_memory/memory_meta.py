from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.pyplot as plt

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "memory_meta"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "memory_meta"
output_dir.mkdir(exist_ok=True, parents=True)

df_0 = pd.read_csv(Path("output") / "k562_memory_bench" / "k562_memory_results.csv")
df_0["dataset"] = "k562"

df_1 = pd.read_csv(Path("output") / "hct116_memory_bench" / "hct116_memory_results.csv")
df_1["dataset"] = "hct116"

df_2 = pd.read_csv(Path("output") / "hek293t_memory_bench" / "hek293t_memory_results.csv")
df_2["dataset"] = "hek293t"

df = pd.concat([df_0, df_1, df_2])
df["coreset_fraction"] = pd.Categorical(df["coreset_fraction"])
df.to_csv(output_dir / "memory_meta_benchmark_summary.csv", index=False)

# plotting meta
coreset_colors = {
    1.0: "#a24343",
    0.1: "#96b728",
}

p1 = (
    pn.ggplot(df)
    + pn.geom_point(
        pn.aes(x="n_cells", y="mem_rss_peak_over_start_mb", color="coreset_fraction"),
        alpha=0.5
    )
    + pn.facet_grid(rows="dataset", cols="n_archetypes")
    + pn.scale_x_log10()
    + pn.scale_color_manual(values=coreset_colors, name="Coreset\nFraction")
    + pn.theme_minimal()
)
p1.save(figure_dir / "memory_usage_memory_meta.pdf", dpi=300, width=8, height=6)
p1.save(figure_dir / "memory_usage_memory_meta.png", dpi=300, width=8, height=6)

p2 = (
    pn.ggplot(df)
    + pn.geom_point(
        pn.aes(x="n_cells", y="time_compute", color="coreset_fraction"),
        alpha=0.5
    )
    + pn.facet_grid(rows="dataset", cols="n_archetypes")
    + pn.scale_x_log10()
    + pn.scale_color_manual(values=coreset_colors, name="Coreset\nFraction")
    + pn.theme_minimal()
)
p2.save(figure_dir / "computation_time_memory_meta.pdf", dpi=300, width=8, height=6)
p2.save(figure_dir / "computation_time_memory_meta.png", dpi=300, width=8, height=6)
