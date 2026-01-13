from pathlib import Path

import matplotlib
import scanpy as sc
import numpy as np

from ..utils.const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

project_path = Path(".")

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_bench"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_bench"
output_dir.mkdir(exist_ok=True, parents=True)

# read the data
adata = sc.read_h5ad(
    project_path
    / ".."
    / "gpp_bench"
    / "data"
    / "prc"
    / "orion_hct116"
    / "orion_hct116_adata.h5ad"
)

# subsample the data
n_cells_max = 1_000_000
random_state = 42

rng = np.random.default_rng(random_state)
n_obs = adata.n_obs
if n_obs > n_cells_max:
    idx = rng.choice(n_obs, size=n_cells_max, replace=False)
    adata = adata[idx].copy()

# process the data
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
#sc.pp.pca(adata, mask_var="highly_variable", n_comps=50)?
# sc.pp.pca(adata, mask_var="highly_variable", n_comps=50)?

n_archetypes_list = [2, 3, 4, 5, 6]
n_cells_list = [1*1e3, 2*1e3, 10*1e3, 20*1e3, 100*1e3, 200*1e3, 1_000*1e3]

seed_list = SEED_DICT["s"]

result_dict = {}

for n_archetypes in n_archetypes_list:
    for n_cells in n_cells_list:
        if n_cells != n_cells_max:
            idx = rng.choice(n_obs, size=n_cells, replace=False)
            adata_with_n_cells = adata = adata[idx].copy()
        else:
            adata_with_n_cells = adata
        for seed in seed_list:
            # TODO: Benchmark memory and wall clock time
            pass
            