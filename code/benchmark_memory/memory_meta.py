from pathlib import Path

import pandas as pd
import plotnine as pn

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

df_2 = pd.read_csv(
    Path("output") / "hek293t_memory_bench" / "hek293t_memory_results.csv"
)
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
        alpha=0.5,
    )
    + pn.facet_grid(rows="dataset", cols="n_archetypes")
    + pn.scale_x_log10()
    + pn.scale_color_manual(values=coreset_colors, name="Coreset\nFraction")
    + pn.theme_bw()
    + pn.labs(x="Number of Cells", y="Peak Memory Usage (MB)")
    + pn.theme(strip_background=pn.element_rect(fill="none", color="none"))
)
p1.save(figure_dir / "memory_usage_memory_meta.pdf", dpi=300, width=8, height=6)
p1.save(figure_dir / "memory_usage_memory_meta.png", dpi=300, width=8, height=6)

p2 = p1 + pn.scale_y_log10()
p2.save(figure_dir / "memory_usage_memory_meta_logy.pdf", dpi=300, width=8, height=6)
p2.save(figure_dir / "memory_usage_memory_meta_logy.png", dpi=300, width=8, height=6)

p3 = (
    pn.ggplot(df)
    + pn.geom_point(
        pn.aes(x="n_cells", y="time_compute", color="coreset_fraction"), alpha=0.5
    )
    + pn.facet_grid(rows="dataset", cols="n_archetypes")
    + pn.scale_x_log10()
    + pn.scale_color_manual(values=coreset_colors, name="Coreset\nFraction")
    + pn.theme_bw()
    + pn.labs(x="Number of Cells", y="Computation Time (s)")
    + pn.theme(strip_background=pn.element_rect(fill="none", color="none"))
)
p3.save(figure_dir / "computation_time_memory_meta.pdf", dpi=300, width=8, height=6)
p3.save(figure_dir / "computation_time_memory_meta.png", dpi=300, width=8, height=6)

p4 = p3 + pn.scale_y_log10()
p4.save(
    figure_dir / "computation_time_memory_meta_logy.pdf", dpi=300, width=8, height=6
)
p4.save(
    figure_dir / "computation_time_memory_meta_logy.png", dpi=300, width=8, height=6
)
