#!/usr/bin/env python3
"""
run_simulations.py

Runs the actual multi-fidelity HPO simulation study on one model size's DPO-AO
blackbox (built by build_blackbox.py).

For every (search objective, optimizer, seed) triple, we simulate a full HPO
run with Syne Tune's simulator backend (``UserBlackboxBackend`` +
``SimulatorCallback``): the "training" cost of every (config, checkpoint) is
looked up from the table rather than executed, but the tuner still behaves as
if it were running on 2 real GPU workers with the DPO training's real elapsed
time -- so ASHA/BOHB's early-stopping decisions translate into real simulated
wall-clock savings.

Optimizers compared (a 2x2 of model-free/model-based x single/multi-fidelity):
  - RandomSearch : model-free,  single-fidelity (always trains to completion)
  - BOTorch      : GP-based BO, single-fidelity (always trains to completion)
  - ASHA         : model-free,  multi-fidelity   (early-stops weak configs)
  - BOHB         : KDE-based BO, multi-fidelity   (early-stops weak configs)

Search objectives compared: ELO, Arena-Hard, MT-Bench, AlpacaEval (four
different pairwise-judge signals) and Z-All (the composite z-score across all
11 benchmarks, used as a proxy "ground truth" for overall quality). Since the
blackbox always reports *every* objective at every fidelity regardless of
which one is being searched, the resulting per-trial table lets us later ask,
for a run that searched for e.g. ELO: "how good was the config it found,
according to Arena-Hard / GSM8K / Z-All / ...?"

25 seeds x 4 optimizers x 5 objectives = 500 simulated tuning runs per size.
Each is independent and fast (pure table lookups + a small GP/KDE fit), so
this completes in a few minutes per size.

Usage:
    python run_simulations.py --size 1.7b
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

RESULTS_ROOT = None  # set before importing syne_tune so SYNETUNE_FOLDER is honored


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    return parser.parse_args()


_args = _parse_args()


def _setup_results_root(size_key: str, budget_tag: str) -> None:
    global RESULTS_ROOT
    from common import get_paths

    paths = get_paths(size_key, budget_tag)
    RESULTS_ROOT = paths.results_dir / "syne_tune_runs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["SYNETUNE_FOLDER"] = str(RESULTS_ROOT)


_setup_results_root(_args.size, _args.budget_tag)

import pandas as pd  # noqa: E402
from syne_tune import StoppingCriterion, Tuner  # noqa: E402
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback  # noqa: E402
from syne_tune.blackbox_repository.blackbox_tabular import deserialize  # noqa: E402
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend  # noqa: E402
from syne_tune.experiments import load_experiment  # noqa: E402
from common import (  # noqa: E402
    BUDGET_REGIMES,
    FIDELITY_ATTR,
    N_SEEDS,
    N_WORKERS,
    OPTIMIZERS,
    SEARCH_OBJECTIVES,
    TIME_OBJECTIVE,
    get_paths,
)
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
    size_key = _args.size
    budget_tag = _args.budget_tag
    paths = get_paths(size_key, budget_tag)
    blackbox = deserialize(str(paths.blackbox_dir))[f"qwen3-{size_key}-dpo-ao"]
    max_t = int(blackbox.fidelity_values.max())
    min_t = int(blackbox.fidelity_values.min())

    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    full_run_cost = float(blackbox.objectives_evaluations[0, 0, -1, time_idx])
    budget = BUDGET_REGIMES[budget_tag] * full_run_cost
    print(f"Size {size_key}, budget regime {budget_tag!r} ({BUDGET_REGIMES[budget_tag]}x): "
          f"full-fidelity run cost ~{full_run_cost:.0f}s -> budget {budget:.0f}s/run")

    # Not every objective survived build_blackbox.py's data-quality filtering for
    # every size (e.g. AlpacaEval was dropped entirely for 14B - missing for
    # more than half its DPO configs). Search only what this size's blackbox
    # actually reports.
    objectives = [o for o in SEARCH_OBJECTIVES if o in blackbox.objectives_names]
    skipped = [o for o in SEARCH_OBJECTIVES if o not in objectives]
    if skipped:
        print(f"[WARN] {size_key}: skipping search objectives not available for this size: {skipped}")

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
        "config_lr", "config_beta", FIDELITY_ATTR, "st_tuner_time",
    ] + blackbox.objectives_names  # whatever this size's blackbox actually reports
    raw = raw[keep_cols].rename(columns={"config_lr": "lr", "config_beta": "beta"})

    paths.simulation_raw_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(paths.simulation_raw_csv, index=False)
    print(f"Saved raw per-report results: {paths.simulation_raw_csv} ({len(raw)} rows)")

    # "Best found": for each (objective, optimizer, seed) run, the config/report
    # that achieved the highest value of the searched objective, and everything
    # else it scored at that same report (used for cross-metric transfer).
    idx = raw.groupby(["search_objective", "optimizer", "seed"]).apply(lambda g: g[g.name[0]].idxmax())
    best = raw.loc[idx.values].reset_index(drop=True)
    best.to_csv(paths.best_found_csv, index=False)
    print(f"Saved best-found-per-run summary: {paths.best_found_csv} ({len(best)} rows)")


if __name__ == "__main__":
    main()
