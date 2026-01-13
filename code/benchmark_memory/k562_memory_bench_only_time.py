from pathlib import Path
import logging
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib
import pandas as pd
import scanpy as sc
import numpy as np
import partipy as pt

from ..utils.const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

project_path = Path(".")

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "ms_bench"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "ms_bench"
output_dir.mkdir(exist_ok=True, parents=True)


_MP_ADATA = None


def _benchmark_worker(result_queue, n_archetypes, seed):
    try:
        total_start_time = time.time()
        copy_start_time = time.time()
        adata_bench = _MP_ADATA.copy()
        copy_end_time = time.time()

        compute_start_time = time.time()
        pt.compute_archetypes(
            adata_bench,
            n_archetypes=n_archetypes,
            n_restarts=1,
            seed=seed,
            save_to_anndata=True,
            archetypes_only=False,
            verbose=False,
        )
        compute_end_time = time.time()

        result_queue.put(
            {
                "ok": True,
                "time_total": compute_end_time - total_start_time,
                "time_copy": copy_end_time - copy_start_time,
                "time_compute": compute_end_time - compute_start_time,
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(),
            }
        )


def setup_logger(log_path, level=logging.INFO):
    logger = logging.getLogger("k562_runtime_bench")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger(output_dir / "k562_runtime_bench.log")

logger.info("Starting K562 runtime benchmark")
logger.info(f"Output dir: {output_dir.resolve()}")
logger.info(f"Figure dir: {figure_dir.resolve()}")

# read the data
data_path = (
    project_path
    / ".."
    / "gpp_bench"
    / "data"
    / "prc"
    / "replogle2022_k562_gwps"
    / "replogle2022_k562_gwps_adata.h5ad"
)
logger.info(f"Reading data: {data_path}")
adata = sc.read_h5ad(data_path)
logger.info(f"Loaded data: n_obs={adata.n_obs} n_vars={adata.n_vars}")

# settings
# seed_list = SEED_DICT["s"]
seed_list = SEED_DICT["xs"]
n_archetypes_list = [2, 3, 4, 5, 6]
# n_cells_list = [1*1e3, 2*1e3, 10*1e3, 20*1e3, 100*1e3, 200*1e3, 1_000*1e3]
n_cells_list = [1_000, 2_000, 10_000, 20_000]

# subsample the data
n_cells_max = max(n_cells_list)
random_state = 42

rng = np.random.default_rng(random_state)
n_obs = adata.n_obs
if n_obs > n_cells_max:
    idx = rng.choice(n_obs, size=n_cells_max, replace=False)
    adata = adata[idx].copy()
    logger.info(f"Subsampled to n_obs={adata.n_obs} (max={n_cells_max})")
else:
    logger.info(f"No subsampling needed (n_obs={n_obs} <= max={n_cells_max})")
n_obs = adata.n_obs
idx_order = np.random.default_rng(random_state).permutation(n_obs)
idx_per_n_cells = {n_cells: idx_order[:n_cells] for n_cells in n_cells_list}

# process the data
logger.info("Running preprocessing: normalize_total, log1p, highly_variable_genes")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.pp.pca(adata, mask_var="highly_variable", n_comps=20)
pt.set_obsm(adata=adata, obsm_key="X_pca", n_dimensions=20)

logger.info(f"Benchmark settings: n_archetypes={n_archetypes_list}")
logger.info(f"Benchmark settings: n_cells={n_cells_list}")
logger.info(f"Benchmark settings: seeds={seed_list}")

worker_timeout_s = 60 * 60

result_rows = []
mp_ctx = mp.get_context("fork")
for n_archetypes in n_archetypes_list:
    logger.info(f"Benchmarking n_archetypes={n_archetypes}")
    for n_cells in n_cells_list:
        if n_cells != n_cells_max:
            idx = idx_per_n_cells[n_cells]
            adata_with_n_cells = adata[idx]
            logger.info(f"Using n_cells={n_cells} (subsampled)")
        else:
            adata_with_n_cells = adata
            logger.info(f"Using n_cells={n_cells} (full)")
        _MP_ADATA = adata_with_n_cells
        for seed in seed_list:
            logger.info(f"Benchmarking seed={seed}")
            result_queue = mp_ctx.Queue()
            proc = mp_ctx.Process(
                target=_benchmark_worker,
                args=(result_queue, n_archetypes, seed),
            )
            wall_start_time = time.time()
            proc.start()
            try:
                result = result_queue.get(timeout=worker_timeout_s)
            except queue.Empty:
                wall_elapsed = time.time() - wall_start_time
                logger.error(
                    f"Timeout: n_archetypes={n_archetypes} n_cells={n_cells} "
                    f"seed={seed} after {worker_timeout_s}s (wall={wall_elapsed:.2f}s)"
                )
                proc.terminate()
                proc.join()
                result_rows.append(
                    {
                        "n_archetypes": n_archetypes,
                        "n_cells": n_cells,
                        "seed": seed,
                        "ok": False,
                        "error": f"timeout after {worker_timeout_s}s",
                        "wall_time": wall_elapsed,
                        "time": np.nan,
                        "time_total": np.nan,
                        "time_copy": np.nan,
                        "time_compute": np.nan,
                    }
                )
                continue
            proc.join()
            wall_elapsed = time.time() - wall_start_time
            if proc.exitcode != 0:
                logger.error(
                    f"Worker failed: n_archetypes={n_archetypes} n_cells={n_cells} "
                    f"seed={seed} exitcode={proc.exitcode} wall={wall_elapsed:.2f}s"
                )
                result_rows.append(
                    {
                        "n_archetypes": n_archetypes,
                        "n_cells": n_cells,
                        "seed": seed,
                        "ok": False,
                        "error": f"worker exit code {proc.exitcode}",
                        "wall_time": wall_elapsed,
                        "time": np.nan,
                        "time_total": np.nan,
                        "time_copy": np.nan,
                        "time_compute": np.nan,
                    }
                )
                continue
            if not result["ok"]:
                logger.error(
                    f"Worker error: n_archetypes={n_archetypes} n_cells={n_cells} "
                    f"seed={seed} error={result['error']} wall={wall_elapsed:.2f}s"
                )
                result_rows.append(
                    {
                        "n_archetypes": n_archetypes,
                        "n_cells": n_cells,
                        "seed": seed,
                        "ok": False,
                        "error": result["error"],
                        "wall_time": wall_elapsed,
                        "time": np.nan,
                        "time_total": np.nan,
                        "time_copy": np.nan,
                        "time_compute": np.nan,
                    }
                )
                continue

            result_row = {
                "n_archetypes": n_archetypes,
                "n_cells": n_cells,
                "seed": seed,
                "ok": True,
                "error": "",
                "wall_time": wall_elapsed,
                "time": result["time_compute"],
                "time_total": result["time_total"],
                "time_copy": result["time_copy"],
                "time_compute": result["time_compute"],
            }
            result_rows.append(result_row)
            logger.info(
                f"Run complete: n_archetypes={n_archetypes} n_cells={n_cells} "
                f"seed={seed} wall={wall_elapsed:.2f}s "
                f"time_total={result['time_total']:.2f}s "
                f"time_compute={result['time_compute']:.2f}s"
            )

result_df = pd.DataFrame(result_rows)
results_path = output_dir / "k562_runtime_results.csv"
result_df.to_csv(results_path, index=False)
logger.info(f"Wrote results to {results_path.resolve()}")
