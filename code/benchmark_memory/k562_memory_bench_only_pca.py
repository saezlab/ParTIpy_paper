from pathlib import Path
import logging
import multiprocessing as mp
import os
import queue
import resource
import sys
import threading
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
from ..utils.data_utils import guess_is_lognorm

# for notebooks:
#from pathlib import Path
#import sys
##repo_root = Path("/home/pschaefer/sds-hd/sd22b002/projects/ParTIpy_paper") # beast
#repo_root = Path("/mnt/sds-hd/sd22b002/projects/ParTIpy_paper") # helix
#if str(repo_root) not in sys.path:
#    sys.path.insert(0, str(repo_root))
#sys.modules.pop("code", None)
#from code.utils.data_utils import load_ms_data
#from code.utils.const import FIGURE_PATH, OUTPUT_PATH, SEED_DICT

project_path = Path(".")

## set up backend for matplotlib: https://matplotlib.org/stable/users/explain/figure/backends.html
matplotlib.use("Agg")

## set up output directory
figure_dir = Path(FIGURE_PATH) / "k562_memory_bench_only_pca"
figure_dir.mkdir(exist_ok=True, parents=True)

output_dir = Path(OUTPUT_PATH) / "k562_memory_bench_only_pca"
output_dir.mkdir(exist_ok=True, parents=True)


def get_rss_mb():
    """Current resident set size (RSS) in MB (Linux /proc)."""
    with open("/proc/self/statm", "r", encoding="utf-8") as handle:
        rss_pages = int(handle.read().split()[1])
    return rss_pages * os.sysconf("SC_PAGE_SIZE") / (1024**2)


def get_peak_rss_mb():
    """Peak RSS in MB since process start (Linux ru_maxrss in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# NOTE: Here, we will only assign the PCA embedding used for AA to a global variable
_MP_PCA = None

RSS_POLL_MS = 50


def _rss_sampler(stop_event, interval_s, result_dict, result_key):
    max_rss = get_rss_mb()
    while not stop_event.is_set():
        rss = get_rss_mb()
        if rss > max_rss:
            max_rss = rss
        stop_event.wait(interval_s)
    result_dict[result_key] = max_rss


def _benchmark_worker(result_queue, n_archetypes, seed, coreset_fraction):
    try:
        rss_total_start_mb = get_rss_mb()
        peak_total_start_mb = get_peak_rss_mb()
        total_start_time = time.time()
        rss_poll_s = RSS_POLL_MS / 1000.0
        sampled_peaks = {}

        copy_stop = threading.Event()
        copy_thread = threading.Thread(
            target=_rss_sampler,
            args=(copy_stop, rss_poll_s, sampled_peaks, "copy"),
        )
        copy_thread.start()
        copy_start_time = time.time()
        pca_bench = _MP_PCA.copy()
        copy_end_time = time.time()
        copy_stop.set()
        copy_thread.join()
        rss_post_copy_mb = get_rss_mb()
        peak_post_copy_mb = get_peak_rss_mb()

        compute_stop = threading.Event()
        compute_thread = threading.Thread(
            target=_rss_sampler,
            args=(compute_stop, rss_poll_s, sampled_peaks, "compute"),
        )
        compute_thread.start()
        compute_start_time = time.time()
        coreset_kwargs = {}
        if coreset_fraction < 1.0:
            coreset_kwargs["coreset_algorithm"] = "standard"
            coreset_kwargs["coreset_fraction"] = coreset_fraction

        aa_model = pt.AA(n_archetypes=n_archetypes, verbose=False, seed=seed, **coreset_kwargs)
        aa_model.fit(X=pca_bench)

        compute_end_time = time.time()
        compute_stop.set()
        compute_thread.join()
        rss_end_mb = get_rss_mb()
        peak_end_mb = get_peak_rss_mb()

        result_queue.put(
            {
                "ok": True,
                "rss_poll_ms": RSS_POLL_MS,
                "time_total": compute_end_time - total_start_time,
                "time_copy": copy_end_time - copy_start_time,
                "time_compute": compute_end_time - compute_start_time,
                "mem_rss_start_mb": rss_total_start_mb,
                "mem_rss_post_copy_mb": rss_post_copy_mb,
                "mem_rss_end_mb": rss_end_mb,
                "mem_rss_delta_post_copy_mb": rss_end_mb - rss_post_copy_mb,
                "mem_rss_delta_since_start_mb": rss_end_mb - rss_total_start_mb,
                "mem_rss_peak_mb": peak_end_mb,
                "mem_rss_peak_over_post_copy_mb": peak_end_mb - peak_post_copy_mb,
                "mem_rss_peak_over_start_mb": peak_end_mb - peak_total_start_mb,
                "mem_rss_peak_copy_sampled_mb": sampled_peaks.get("copy", np.nan),
                "mem_rss_peak_compute_sampled_mb": sampled_peaks.get("compute", np.nan),
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
    logger = logging.getLogger("k562_memory_bench")
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


logger = setup_logger(output_dir / "k562_memory_bench.log")

logger.info("Starting K562 memory benchmark")
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
assert not guess_is_lognorm(adata)
logger.info(f"Loaded data: n_obs={adata.n_obs} n_vars={adata.n_vars}")

# testing settings
# seed_list = SEED_DICT["s"]
# n_cells_list = [1_000, 2_000, 10_000, 20_000]
# n_archetypes_list = [2, 3]

# actual settings
n_archetypes_list = [3, 6, 9]
seed_list = SEED_DICT["s"]
n_cells_list = [1_000, 2_000, 10_000, 20_000, 100_000, 200_000, 1_000_000]
coreset_fraction_list = [1.00, 0.10]

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
logger.info(f"Benchmark settings: coreset_fractions={coreset_fraction_list}")

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
        _MP_PCA = adata_with_n_cells.obsm["X_pca"][:, :20]
        for coreset_fraction in coreset_fraction_list:
            logger.info(f"Benchmarking coreset_fraction={coreset_fraction:.2f}")
            for seed in seed_list:
                logger.info(
                    "Benchmarking seed=%s (coreset_fraction=%.2f)",
                    seed,
                    coreset_fraction,
                )
                result_queue = mp_ctx.Queue()
                proc = mp_ctx.Process(
                    target=_benchmark_worker,
                    args=(result_queue, n_archetypes, seed, coreset_fraction),
                )
                proc.start()
                try:
                    result = result_queue.get(timeout=worker_timeout_s)
                except queue.Empty:
                    logger.error(
                        "Timeout: n_archetypes=%s n_cells=%s coreset_fraction=%.2f seed=%s after %ss",
                        n_archetypes,
                        n_cells,
                        coreset_fraction,
                        seed,
                        worker_timeout_s,
                    )
                    proc.terminate()
                    proc.join()
                    result_rows.append(
                        {
                            "n_archetypes": n_archetypes,
                            "n_cells": n_cells,
                            "coreset_fraction": coreset_fraction,
                            "seed": seed,
                            "ok": False,
                            "error": f"timeout after {worker_timeout_s}s",
                            "rss_poll_ms": RSS_POLL_MS,
                            "time": np.nan,
                            "time_total": np.nan,
                            "time_copy": np.nan,
                            "time_compute": np.nan,
                            "mem_rss_start_mb": np.nan,
                            "mem_rss_post_copy_mb": np.nan,
                            "mem_rss_end_mb": np.nan,
                            "mem_rss_delta_post_copy_mb": np.nan,
                            "mem_rss_delta_since_start_mb": np.nan,
                            "mem_rss_peak_mb": np.nan,
                            "mem_rss_peak_over_post_copy_mb": np.nan,
                            "mem_rss_peak_over_start_mb": np.nan,
                            "mem_rss_peak_copy_sampled_mb": np.nan,
                            "mem_rss_peak_compute_sampled_mb": np.nan,
                        }
                    )
                    continue
                proc.join()
                if proc.exitcode != 0:
                    logger.error(
                        "Worker failed: n_archetypes=%s n_cells=%s coreset_fraction=%.2f seed=%s exitcode=%s",
                        n_archetypes,
                        n_cells,
                        coreset_fraction,
                        seed,
                        proc.exitcode,
                    )
                    result_rows.append(
                        {
                            "n_archetypes": n_archetypes,
                            "n_cells": n_cells,
                            "coreset_fraction": coreset_fraction,
                            "seed": seed,
                            "ok": False,
                            "error": f"worker exit code {proc.exitcode}",
                            "rss_poll_ms": RSS_POLL_MS,
                            "time": np.nan,
                            "time_total": np.nan,
                            "time_copy": np.nan,
                            "time_compute": np.nan,
                            "mem_rss_start_mb": np.nan,
                            "mem_rss_post_copy_mb": np.nan,
                            "mem_rss_end_mb": np.nan,
                            "mem_rss_delta_post_copy_mb": np.nan,
                            "mem_rss_delta_since_start_mb": np.nan,
                            "mem_rss_peak_mb": np.nan,
                            "mem_rss_peak_over_post_copy_mb": np.nan,
                            "mem_rss_peak_over_start_mb": np.nan,
                            "mem_rss_peak_copy_sampled_mb": np.nan,
                            "mem_rss_peak_compute_sampled_mb": np.nan,
                        }
                    )
                    continue
                if not result["ok"]:
                    logger.error(
                        "Worker error: n_archetypes=%s n_cells=%s coreset_fraction=%.2f seed=%s error=%s",
                        n_archetypes,
                        n_cells,
                        coreset_fraction,
                        seed,
                        result["error"],
                    )
                    result_rows.append(
                        {
                            "n_archetypes": n_archetypes,
                            "n_cells": n_cells,
                            "coreset_fraction": coreset_fraction,
                            "seed": seed,
                            "ok": False,
                            "error": result["error"],
                            "rss_poll_ms": RSS_POLL_MS,
                            "time": np.nan,
                            "time_total": np.nan,
                            "time_copy": np.nan,
                            "time_compute": np.nan,
                            "mem_rss_start_mb": np.nan,
                            "mem_rss_post_copy_mb": np.nan,
                            "mem_rss_end_mb": np.nan,
                            "mem_rss_delta_post_copy_mb": np.nan,
                            "mem_rss_delta_since_start_mb": np.nan,
                            "mem_rss_peak_mb": np.nan,
                            "mem_rss_peak_over_post_copy_mb": np.nan,
                            "mem_rss_peak_over_start_mb": np.nan,
                            "mem_rss_peak_copy_sampled_mb": np.nan,
                            "mem_rss_peak_compute_sampled_mb": np.nan,
                        }
                    )
                    continue

                result_row = {
                    "n_archetypes": n_archetypes,
                    "n_cells": n_cells,
                    "coreset_fraction": coreset_fraction,
                    "seed": seed,
                    "ok": True,
                    "error": "",
                    "rss_poll_ms": result["rss_poll_ms"],
                    "time": result["time_compute"],
                    "time_total": result["time_total"],
                    "time_copy": result["time_copy"],
                    "time_compute": result["time_compute"],
                    "mem_rss_start_mb": result["mem_rss_start_mb"],
                    "mem_rss_post_copy_mb": result["mem_rss_post_copy_mb"],
                    "mem_rss_end_mb": result["mem_rss_end_mb"],
                    "mem_rss_delta_post_copy_mb": result["mem_rss_delta_post_copy_mb"],
                    "mem_rss_delta_since_start_mb": result[
                        "mem_rss_delta_since_start_mb"
                    ],
                    "mem_rss_peak_mb": result["mem_rss_peak_mb"],
                    "mem_rss_peak_over_post_copy_mb": result[
                        "mem_rss_peak_over_post_copy_mb"
                    ],
                    "mem_rss_peak_over_start_mb": result["mem_rss_peak_over_start_mb"],
                    "mem_rss_peak_copy_sampled_mb": result[
                        "mem_rss_peak_copy_sampled_mb"
                    ],
                    "mem_rss_peak_compute_sampled_mb": result[
                        "mem_rss_peak_compute_sampled_mb"
                    ],
                }
                result_rows.append(result_row)
                logger.info(
                    "Run complete: n_archetypes=%s n_cells=%s coreset_fraction=%.2f seed=%s time_total=%.2fs time_compute=%.2fs rss_peak=%.2fMB (start=%.2fMB, delta=%.2fMB)",
                    n_archetypes,
                    n_cells,
                    coreset_fraction,
                    seed,
                    result["time_total"],
                    result["time_compute"],
                    result["mem_rss_peak_mb"],
                    result["mem_rss_start_mb"],
                    result["mem_rss_peak_over_start_mb"],
                )

result_df = pd.DataFrame(result_rows)
results_path = output_dir / "k562_memory_results.csv"
result_df.to_csv(results_path, index=False)
logger.info(f"Wrote results to {results_path.resolve()}")
