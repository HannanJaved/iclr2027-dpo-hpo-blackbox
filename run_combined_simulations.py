#!/usr/bin/env python3
"""
run_combined_simulations.py

Runs the joint cross-size HPO simulation study on the combined blackbox built
by build_combined_blackbox.py, where "size" is a searchable hyperparameter
alongside lr and beta (45 configs = 5 sizes x 3 lrs x 3 betas). Unlike
run_simulations.py (one independent search per size), this is a single search
problem spanning all 5 sizes at once, testing whether an optimizer correctly
trades off a bigger model's ~6.7x higher training cost against its usually
(but not always) higher achievable quality.

Search objectives: ELO (one shared tournament pool across all sizes, so
directly comparable) and Z-All (recomputed pooled across all 45 configs by
build_combined_blackbox.py, so it also reflects absolute cross-size strength
rather than per-size standing).

Budget: BUDGET_REGIMES multiplier x the AVERAGE full-run cost across the 5
sizes (~30,116s / 8.4h), not any single size's own cost -- since size is now
a decision variable, there's no one "this run's" reference cost. This makes
the budget genuinely adversarial for 14B (a single 14B trial, ~68,712s/19h,
can nearly exhaust the "tight" 1x budget on its own) while being generous
relative to 0.6B (~10,290s/2.9h) -- exactly the cost spread the experiment is
meant to probe.

Usage:
    python run_combined_simulations.py [--budget-tag generous|tight]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import time

RESULTS_ROOT = None  # set before importing syne_tune so SYNETUNE_FOLDER is honored


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    return parser.parse_args()


_args = _parse_args()


def _setup_results_root(budget_tag: str) -> None:
    global RESULTS_ROOT
    from common import get_combined_paths

    paths = get_combined_paths(budget_tag)
    RESULTS_ROOT = paths.results_dir / "syne_tune_runs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["SYNETUNE_FOLDER"] = str(RESULTS_ROOT)


_setup_results_root(_args.budget_tag)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from syne_tune import StoppingCriterion, Tuner  # noqa: E402
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback  # noqa: E402
from syne_tune.blackbox_repository.blackbox_tabular import deserialize  # noqa: E402
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend  # noqa: E402
from syne_tune.experiments import load_experiment  # noqa: E402

from common import BUDGET_REGIMES, COMBINED_SEARCH_OBJECTIVES, FIDELITY_ATTR, N_SEEDS, N_WORKERS  # noqa: E402
from common import OPTIMIZERS, TIME_OBJECTIVE, get_combined_paths  # noqa: E402
from schedulers import make_scheduler  # noqa: E402


def run_one(blackbox, objective: str, optimizer: str, seed: int, max_t: int, min_t: int, budget: float) -> pd.DataFrame:
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


def main() -> None:
    budget_tag = _args.budget_tag
    paths = get_combined_paths(budget_tag)
    blackbox = deserialize(str(paths.blackbox_dir))["qwen3-combined-dpo-ao"]
    max_t = int(blackbox.fidelity_values.max())
    min_t = int(blackbox.fidelity_values.min())

    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    avg_full_run_cost = float(np.mean(blackbox.objectives_evaluations[:, 0, -1, time_idx]))
    budget = BUDGET_REGIMES[budget_tag] * avg_full_run_cost
    print(f"Combined, budget regime {budget_tag!r} ({BUDGET_REGIMES[budget_tag]}x): "
          f"average full-fidelity run cost ~{avg_full_run_cost:.0f}s -> budget {budget:.0f}s/run")

    objectives = [o for o in COMBINED_SEARCH_OBJECTIVES if o in blackbox.objectives_names]
    runs = [
        (objective, optimizer, seed)
        for objective in objectives
        for optimizer in OPTIMIZERS
        for seed in range(N_SEEDS)
    ]
    print(f"Running {len(runs)} simulations "
          f"({len(objectives)} objectives x {len(OPTIMIZERS)} optimizers x {N_SEEDS} seeds)")

    all_frames = []
    t0 = time.time()
    n_failed = 0
    for i, (objective, optimizer, seed) in enumerate(runs, 1):
        try:
            all_frames.append(run_one(blackbox, objective, optimizer, seed, max_t, min_t, budget))
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"[WARN] run {objective}/{optimizer}/seed={seed} failed: {exc}")
        if i % 25 == 0 or i == len(runs):
            elapsed = time.time() - t0
            print(f"  {i}/{len(runs)} done ({elapsed:.0f}s elapsed, {n_failed} failed)", flush=True)

    raw = pd.concat(all_frames, ignore_index=True)
    keep_cols = [
        "search_objective", "optimizer", "seed", "trial_id",
        "config_size", "config_lr", "config_beta", FIDELITY_ATTR, "st_tuner_time",
    ] + blackbox.objectives_names
    raw = raw[keep_cols].rename(columns={"config_size": "size", "config_lr": "lr", "config_beta": "beta"})

    paths.simulation_raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(paths.simulation_raw_csv, index=False)
    print(f"Saved raw per-report results: {paths.simulation_raw_csv} ({len(raw)} rows)")

    idx = raw.groupby(["search_objective", "optimizer", "seed"]).apply(lambda g: g[g.name[0]].idxmax())
    best = raw.loc[idx.values].reset_index(drop=True)
    best.to_csv(paths.best_found_csv, index=False)
    print(f"Saved best-found-per-run summary: {paths.best_found_csv} ({len(best)} rows)")


if __name__ == "__main__":
    main()
