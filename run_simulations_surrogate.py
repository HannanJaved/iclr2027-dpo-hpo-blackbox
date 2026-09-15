#!/usr/bin/env python3
"""
run_simulations_surrogate.py

Continuous-space counterpart to run_simulations.py -- same simulated multi-
fidelity HPO study, but against a surrogate blackbox (build_blackbox_surrogate.py,
--surrogate knn1/knn5/gp) instead of the discrete tabular one, so
RandomSearch/TPE/CQR can propose any (lr, beta) in the continuous log-uniform
search space bounded to the observed grid's range, not just the exact
measured points. ASHA/BOHB's fidelity ladder (dpo_step) is unchanged and
still discrete.

Parameters (optimizers, search objectives, N_SEEDS, N_WORKERS, budget
regimes) are the SAME as run_simulations.py/common.py, unchanged, for
comparability with the discrete-grid study -- see that module for their
meaning. In particular N_WORKERS=2 is a simulated-GPU-worker modeling
parameter, not a speed knob; it's deliberately not touched here.

Unlike run_simulations.py, results need no post-hoc cfg_id -> (lr, beta)
decode: the surrogate's hyperparameters already ARE literal (log-space)
lr/beta columns. The Tuner's own "config_log_lr"/"config_log_beta" result
columns are converted back to real lr/beta (10**x) before saving, so the
output CSV schema matches the discrete study's.

Engineering-only change from run_simulations.py: the outer loop over
(objective, optimizer, seed) combinations -- independent, CPU-only, and
previously run one at a time in a single process -- is now parallelized
across a process pool (default: all available CPUs), since run_size.sbatch-
style jobs already reserve --cpus-per-task=16 that a serial loop never used.
Each worker loads its own copy of the (picklable) surrogate blackbox once at
startup rather than re-pickling it per task.

Usage:
    python run_simulations_surrogate.py --size 4b --surrogate gp
    python run_simulations_surrogate.py --size 8b --surrogate knn5 --n-procs 16
"""
from __future__ import annotations

import argparse
import contextlib
import io
import multiprocessing
import os
import pickle
import time

RESULTS_ROOT = None  # set before importing syne_tune so SYNETUNE_FOLDER is honored


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default="qwen3", choices=["qwen3", "llama"])
    parser.add_argument("--surrogate", default="gp", choices=["knn1", "knn3", "knn5", "gp"])
    parser.add_argument("--n-procs", type=int, default=None,
                         help="Worker processes for the outer simulation loop (default: os.cpu_count()). "
                              "Unrelated to N_WORKERS (simulated GPU workers per run, see common.py).")
    return parser.parse_args()


_args = _parse_args()


def _setup_results_root(size_key: str, family: str, surrogate: str) -> dict:
    global RESULTS_ROOT
    from build_blackbox_surrogate import get_surrogate_results_paths

    paths = get_surrogate_results_paths(size_key, family=family, surrogate=surrogate)
    RESULTS_ROOT = paths["results_dir"] / "syne_tune_runs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["SYNETUNE_FOLDER"] = str(RESULTS_ROOT)
    return paths


_paths = _setup_results_root(_args.size, _args.family, _args.surrogate)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from syne_tune import StoppingCriterion, Tuner  # noqa: E402
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback  # noqa: E402
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend  # noqa: E402
from syne_tune.experiments import load_experiment  # noqa: E402
from build_blackbox_surrogate import surrogate_path  # noqa: E402
from common import BUDGET_REGIMES, FIDELITY_ATTR, N_SEEDS, N_WORKERS, OPTIMIZERS, SEARCH_OBJECTIVES, TIME_OBJECTIVE, get_paths  # noqa: E402
from schedulers import make_scheduler  # noqa: E402

_WORKER_BLACKBOX = None  # set once per worker process by _worker_init


def _worker_init(blackbox_path: str) -> None:
    global _WORKER_BLACKBOX
    with open(blackbox_path, "rb") as fh:
        _WORKER_BLACKBOX = pickle.load(fh)


def run_one(objective: str, optimizer: str, seed: int, max_t: int, min_t: int, budget: float) -> pd.DataFrame:
    blackbox = _WORKER_BLACKBOX
    scheduler = make_scheduler(optimizer, blackbox.configuration_space, objective, max_t, min_t, seed)
    backend = UserBlackboxBackend(blackbox=blackbox, elapsed_time_attr=TIME_OBJECTIVE)
    tuner_name = f"{objective}-{optimizer}-s{seed}"[:60].replace(" ", "")
    tuner = Tuner(
        trial_backend=backend,
        scheduler=scheduler,
        stop_criterion=StoppingCriterion(max_wallclock_time=budget),
        n_workers=N_WORKERS,
        sleep_time=0,
        callbacks=[SimulatorCallback()],
        tuner_name=tuner_name,
        suffix_tuner_name=True,
        save_tuner=False,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        tuner.run()
    exp = load_experiment(tuner.name)
    df = exp.results.copy()
    df["search_objective"] = objective
    df["optimizer"] = optimizer
    df["seed"] = seed
    return df


def _run_one_task(task: tuple[str, str, int, int, int, float]) -> tuple[str, str, int, pd.DataFrame | None, str | None]:
    objective, optimizer, seed, max_t, min_t, budget = task
    try:
        return objective, optimizer, seed, run_one(objective, optimizer, seed, max_t, min_t, budget), None
    except Exception as exc:  # noqa: BLE001
        return objective, optimizer, seed, None, str(exc)


def main() -> None:
    size_key = _args.size
    family = _args.family
    n_procs = _args.n_procs or os.cpu_count()

    surrogate = _args.surrogate
    blackbox_path = str(surrogate_path(size_key, family, surrogate))
    with open(blackbox_path, "rb") as fh:
        blackbox = pickle.load(fh)
    max_t = int(blackbox.fidelity_values.max())
    min_t = int(blackbox.fidelity_values.min())

    # Unlike BlackboxTabular, BlackboxSurrogate keeps no .hyperparameters
    # table (the raw X/y only went into fitting the sklearn pipeline) -- get
    # a real (lr, beta) point straight from the grid CSV instead. Configs are
    # fit/searched in log10 space (see build_blackbox_surrogate.py).
    grid_df = pd.read_csv(get_paths(size_key, family=family).grid_csv)
    real_point = {"log_lr": float(np.log10(grid_df["lr"].iloc[0])), "log_beta": float(np.log10(grid_df["beta"].iloc[0]))}

    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    # Full-run cost at a real grid point (surrogate is exact there, see
    # build_blackbox_surrogate.py's on-grid sanity check) -- same convention
    # as run_simulations.py.
    curve = blackbox.objective_function(real_point)
    full_run_cost = float(curve[-1, time_idx])
    budget = BUDGET_REGIMES["generous"] * full_run_cost
    print(f"{family}-{size_key} (surrogate), budget regime 'generous' ({BUDGET_REGIMES['generous']}x): "
          f"full-fidelity run cost ~{full_run_cost:.0f}s -> budget {budget:.0f}s/run")

    objectives = [o for o in SEARCH_OBJECTIVES if o in blackbox.objectives_names]
    skipped = [o for o in SEARCH_OBJECTIVES if o not in objectives]
    if skipped:
        print(f"[WARN] {size_key}: skipping search objectives not available for this size: {skipped}")

    tasks = [
        (objective, optimizer, seed, max_t, min_t, budget)
        for objective in objectives
        for optimizer in OPTIMIZERS
        for seed in range(N_SEEDS)
    ]
    print(f"Running {len(tasks)} simulations "
          f"({len(objectives)} objectives x {len(OPTIMIZERS)} optimizers x {N_SEEDS} seeds) "
          f"across {n_procs} worker processes")

    all_frames = []
    t0 = time.time()
    n_failed = 0
    with multiprocessing.Pool(processes=n_procs, initializer=_worker_init, initargs=(blackbox_path,)) as pool:
        for i, (objective, optimizer, seed, df, err) in enumerate(pool.imap_unordered(_run_one_task, tasks), 1):
            if err is not None:
                n_failed += 1
                print(f"[WARN] run {objective}/{optimizer}/seed={seed} failed: {err}")
            else:
                all_frames.append(df)
            if i % 25 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  {i}/{len(tasks)} done ({elapsed:.0f}s elapsed, {n_failed} failed)", flush=True)

    raw = pd.concat(all_frames, ignore_index=True)
    keep_cols = [
        "search_objective", "optimizer", "seed", "trial_id",
        "config_log_lr", "config_log_beta", FIDELITY_ATTR, "st_tuner_time",
    ] + blackbox.objectives_names
    raw = raw[keep_cols].rename(columns={"config_log_lr": "log_lr", "config_log_beta": "log_beta"})
    # Convert back to real lr/beta for the saved CSV -- downstream consumers
    # (analyze_results_surrogate.py included) and the earlier knn1 run's
    # output both use plain "lr"/"beta"; only the search/fit machinery needs
    # log-space.
    raw["lr"] = 10 ** raw["log_lr"]
    raw["beta"] = 10 ** raw["log_beta"]
    raw = raw.drop(columns=["log_lr", "log_beta"])

    _paths["results_dir"].mkdir(parents=True, exist_ok=True)
    raw.to_csv(_paths["simulation_raw_csv"], index=False)
    print(f"Saved raw per-report results: {_paths['simulation_raw_csv']} ({len(raw)} rows)")

    idx = raw.groupby(["search_objective", "optimizer", "seed"]).apply(lambda g: g[g.name[0]].idxmax())
    best = raw.loc[idx.values].reset_index(drop=True)
    best.to_csv(_paths["best_found_csv"], index=False)
    print(f"Saved best-found-per-run summary: {_paths['best_found_csv']} ({len(best)} rows)")


if __name__ == "__main__":
    main()
