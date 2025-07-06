from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as pn

from ..utils.const import FIGURE_PATH, OUTPUT_PATH

## set up output directory
figure_dir = Path(FIGURE_PATH) / "coreset_meta"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "coreset_meta"
output_dir.mkdir(exist_ok=True, parents=True)

df_0_meta = pd.read_csv(Path("output") / "ms_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_0 = pd.read_csv(Path("output") / "ms_coreset" / "time_savings.csv")
assert df_0.shape[0] == df_0_meta.shape[0]
df_0["dataset"] = "ms"
df_0 = df_0.join(df_0_meta.set_index("celltype"), on="celltype", how="left")

df_1_meta = pd.read_csv(Path("output") / "lupus_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_1 = pd.read_csv(Path("output") / "lupus_coreset" / "time_savings.csv")
assert df_1.shape[0] == df_1_meta.shape[0]
df_1["dataset"] = "lupus"
df_1 = df_1.join(df_1_meta.set_index("celltype"), on="celltype", how="left")

df_2_meta = pd.read_csv(Path("output") / "ms_xenium_coreset" / "results.csv")[["celltype", "n_samples", "n_dimensions", "n_archetypes"]].drop_duplicates()

df_2 = pd.read_csv(Path("output") / "ms_xenium_coreset" / "time_savings.csv")
assert df_2.shape[0] == df_2_meta.shape[0]
df_2["dataset"] = "ms_xenium"
df_2 = df_2.join(df_2_meta.set_index("celltype"), on="celltype", how="left")

df = pd.concat([df_0, df_1, df_2])
df["log10_n_samples"] = np.log10(df["n_samples"])

df.to_csv(output_dir / "summary.csv")

p = (pn.ggplot(df, mapping=pn.aes(x="n_samples", y="time_saving")) 
     + pn.geom_point(pn.aes(color="n_archetypes"), size=5) 
     + pn.geom_smooth(method="lm") 
     + pn.scale_x_log10()
     + pn.labs(x="Number of Cells (log10)", y="Coreset Time Saving compared to Full Dataset", color="Number of\nArchetypes")
     + pn.scale_color_cmap("copper")
     + pn.theme_bw()
     + pn.theme(axis_text=pn.element_text(size=12), axis_title=pn.element_text(size=16))
     )
p.save(figure_dir / "time_saving_vs_number_of_cells.pdf", dpi=300, verbose=False)

p = (pn.ggplot(df, mapping=pn.aes(x="n_samples", y="coreset_size")) 
     + pn.geom_point(pn.aes(color="n_archetypes"), size=5) 
     + pn.geom_smooth(method="lm") 
     + pn.scale_x_log10()
     + pn.scale_y_log10()
     + pn.labs(x="Number of Cells (log10)", y="Minimal Coreset Fraction", color="Number of\nArchetypes")
     + pn.scale_color_cmap("copper")
     + pn.theme_bw()
     + pn.theme(axis_text=pn.element_text(size=12), axis_title=pn.element_text(size=16))
     )
p.save(figure_dir / "minimal_coreset_fraction_vs_number_of_cells.pdf", dpi=300, verbose=False)
