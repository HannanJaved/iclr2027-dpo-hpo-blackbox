#!/usr/bin/env python3
"""
run_pinned_fidelity_simulations.py

The regular ASHA/BOHB comparison (run_simulations.py) uses syne-tune's default
rung geometry (grace_period=min_t=200, reduction_factor=2), which produces
several rungs (200, 400, 800, 1600, max_t) -- a reasonable general-purpose
schedule, but not a direct simulation of the specific claim this study makes:
"one early checkpoint at ~20% of training (step 400) is enough to decide which
configs are worth finishing." This script builds that experiment directly: an
ASHA/BOHB variant with exactly ONE rung, pinned at step 400, so every trial
gets exactly one early-stopping decision (survive step 400 or don't), then
either gets killed or runs to completion with no further checkpoints in
between. This is what a practitioner using the "read at 20%, commit" strategy
from analyze_transfer_across_fidelity.py would use in an *online, asynchronous*
setting (multiple trials running concurrently, decisions made as results
arrive) rather than that script's offline "wait for the full grid, then pick
the argmax" framing -- a different (more realistic, but noisier) way of asking
the same question.

Rung math: syne-tune's ASHA/BOHB (AsynchronousSuccessiveHalving) computes
MAX_RUNGS = floor(log(max_t/grace_period) / log(reduction_factor)) + 1. With
grace_period=400 and reduction_factor=100 (deliberately large), max_t~2000-2031
gives log(2031/400)/log(100) ~ 0.35, so MAX_RUNGS=1: exactly one rung at 400,
verified against syne_tune.optimizer.schedulers.asha.Bracket directly (not
just the docstring's formula) before trusting it here.

Compared: RandomSearch (no early stopping, the reference), ASHA-step400,
BOHB-step400. Search objectives: Z-Dynamic (the signal RQ1 establishes as the
correct one) and Z-All (for contrast, matching the rest of the study). Uses
each size's own already-built blackbox (build_blackbox.py) and native config
grid (not restricted to the shared 3x3 subgrid, for direct comparability with
run_simulations.py's existing regret_curves.png).

Usage:
    python run_pinned_fidelity_simulations.py --size 1.7b
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
    parser.add_argument("--size", required=True)
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    parser.add_argument("--pinned-step", type=int, default=400)
    parser.add_argument(
        "--optimizers", default=None,
        help="Comma-separated subset of PINNED_OPTIMIZERS to run (default: all of them). "
             "e.g. --optimizers RandomSearch,ASHA-min400,BOHB-min400 for just the -min400 "
             "ablation. NOTE: output is a full overwrite of simulation_raw.csv, not a merge -- "
             "running a subset means downstream analyze_pinned_fidelity.py only sees that "
             "subset until a full (all-optimizers) run is done again for this size/budget-tag.",
    )
    return parser.parse_args()


_args = _parse_args()


def _setup_results_root(size_key: str, budget_tag: str) -> None:
    global RESULTS_ROOT
    from common import HERE, slug

    results_dirname = "results_pinned400" if budget_tag == "generous" else f"results_pinned400_{budget_tag}"
    RESULTS_ROOT = HERE / f"qwen3_{slug(size_key)}_dpo" / results_dirname / "syne_tune_runs"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["SYNETUNE_FOLDER"] = str(RESULTS_ROOT)


_setup_results_root(_args.size, _args.budget_tag)

import pandas as pd  # noqa: E402
from syne_tune import StoppingCriterion, Tuner  # noqa: E402
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback  # noqa: E402
from syne_tune.blackbox_repository.blackbox_tabular import deserialize  # noqa: E402
from syne_tune.blackbox_repository.simulated_tabular_backend import UserBlackboxBackend  # noqa: E402
from syne_tune.experiments import load_experiment  # noqa: E402
from syne_tune.optimizer.baselines import ASHA, BOHB, BOTorch, CQR, RandomSearch, TPE  # noqa: E402
from common import BUDGET_REGIMES, FIDELITY_ATTR, N_SEEDS, N_WORKERS, TIME_OBJECTIVE, get_paths  # noqa: E402

PINNED_OBJECTIVES = ["Z-Dynamic", "Z-All"]
# BOTorch/TPE/CQR are single-fidelity (no rung concept -- they always train to
# completion), so "pinning" them at step 400 is a no-op; they're included
# unchanged from the general study, just now measured on Z-Dynamic/Z-All
# specifically (Z-Dynamic wasn't part of run_simulations.py's SEARCH_OBJECTIVES),
# so this sweep gives the full 6-optimizer picture on the objective that matters.
#
# Two ASHA/BOHB variants, both with grace_period=pinned_step (400) so nothing
# gets judged before the checkpoint we've validated as informative -- but they
# differ in what happens after that:
#   -step400 : reduction_factor=100, brackets=1 -> exactly one rung at step
#     400 (verified: [400]). Survive that single check and you run to
#     completion untouched -- this is what run_pinned_fidelity_simulations.py
#     tested originally.
#   -min400  : reduction_factor=2 (the study's normal default), brackets=1 ->
#     rungs at [400, 800, 1600] (verified against the scheduler directly, not
#     just the formula). Nothing is judged before 400, but survivors face
#     further eliminations at 800 and 1600 instead of running to completion
#     unsupervised -- does giving the algorithm more chances to reallocate
#     budget *after* the informative checkpoint help further, or was one
#     well-placed check already enough? (-min400 turned out worse than
#     -step400 -- but it also confounds "checks again" with "gentler cutoff":
#     reduction_factor=2 keeps ~50% of trials at 400, vs -step400's rf=100
#     keeping ~1%, so more trials survive to compete for the same tight
#     budget either way.)
#   -step400-gentle : isolates that confound. Constructed with the SAME
#     reduction_factor=2 (~50% keep-rate) as -min400, but its rung list is
#     truncated immediately after construction (verified directly on the
#     Bracket object, not inferred) down to just the step-400 entry -- so it
#     keeps -min400's gentler cutoff at 400, but never checks again
#     afterward, like -step400. This is what actually isolates "does
#     re-checking after 400 hurt" from "is the 400 cutoff itself too harsh."
PINNED_OPTIMIZERS = [
    "RandomSearch", "BOTorch", "TPE", "CQR",
    "ASHA-step400", "BOHB-step400",
    "ASHA-min400", "BOHB-min400",
    "ASHA-step400-gentle", "BOHB-step400-gentle",
]


def _truncate_to_single_rung(scheduler, pinned_step: int):
    """Keep only the rung at exactly pinned_step, dropping any later ones
    that reduction_factor=2 would otherwise generate -- verified structurally
    (not just by formula) in the exploration that led to this script."""
    for bracket in scheduler.brackets:
        bracket.rungs = [(m, r) for m, r in bracket.rungs if m == pinned_step]
        assert bracket.rungs == [(pinned_step, {})], (
            f"expected a single rung at {pinned_step}, got {bracket.rungs}"
        )
    return scheduler


def make_pinned_scheduler(name: str, config_space: dict, metric: str, max_t: int, pinned_step: int, seed: int):
    if name == "RandomSearch":
        return RandomSearch(config_space=config_space, metrics=[metric], do_minimize=False, random_seed=seed)
    if name == "BOTorch":
        return BOTorch(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "TPE":
        return TPE(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "CQR":
        return CQR(config_space=config_space, metric=metric, do_minimize=False, random_seed=seed)
    if name == "ASHA-step400":
        return ASHA(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=100, brackets=1,
            do_minimize=False, random_seed=seed,
        )
    if name == "BOHB-step400":
        return BOHB(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=100, brackets=1,
            do_minimize=False, random_seed=seed,
        )
    if name == "ASHA-min400":
        return ASHA(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=2, brackets=1,
            do_minimize=False, random_seed=seed,
        )
    if name == "BOHB-min400":
        return BOHB(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=2, brackets=1,
            do_minimize=False, random_seed=seed,
        )
    if name == "ASHA-step400-gentle":
        sched = ASHA(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=2, brackets=1,
            do_minimize=False, random_seed=seed,
        )
        return _truncate_to_single_rung(sched, pinned_step)
    if name == "BOHB-step400-gentle":
        sched = BOHB(
            config_space=config_space, metric=metric, time_attr=FIDELITY_ATTR,
            max_t=max_t, grace_period=pinned_step, reduction_factor=2, brackets=1,
            do_minimize=False, random_seed=seed,
        )
        return _truncate_to_single_rung(sched, pinned_step)
    raise ValueError(name)


def run_one(blackbox, objective: str, optimizer: str, seed: int, max_t: int, pinned_step: int, budget: float) -> pd.DataFrame:
    scheduler = make_pinned_scheduler(optimizer, blackbox.configuration_space, objective, max_t, pinned_step, seed)
    backend = UserBlackboxBackend(blackbox=blackbox, elapsed_time_attr=TIME_OBJECTIVE)
    tuner_name = f"pinned-{objective}-{optimizer}-s{seed}"[:60].replace(" ", "")
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
    pinned_step = _args.pinned_step
    paths = get_paths(size_key, budget_tag)  # blackbox_dir is budget-tag-independent
    blackbox = deserialize(str(paths.blackbox_dir))[f"qwen3-{size_key}-dpo-ao"]
    max_t = int(blackbox.fidelity_values.max())

    assert pinned_step in blackbox.fidelity_values, (
        f"{size_key}: step {pinned_step} is not one of this blackbox's logged fidelities "
        f"{sorted(blackbox.fidelity_values.tolist())}"
    )

    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    full_run_cost = float(blackbox.objectives_evaluations[0, 0, -1, time_idx])
    budget = BUDGET_REGIMES[budget_tag] * full_run_cost
    print(f"Size {size_key}, pinned rung at step {pinned_step}, budget regime {budget_tag!r}: "
          f"full-fidelity run cost ~{full_run_cost:.0f}s -> budget {budget:.0f}s/run")

    objectives = [o for o in PINNED_OBJECTIVES if o in blackbox.objectives_names]
    skipped = [o for o in PINNED_OBJECTIVES if o not in objectives]
    if skipped:
        print(f"[WARN] {size_key}: skipping objectives not available for this size: {skipped}")

    if _args.optimizers:
        requested = [o.strip() for o in _args.optimizers.split(",") if o.strip()]
        unknown = [o for o in requested if o not in PINNED_OPTIMIZERS]
        if unknown:
            raise ValueError(f"unknown optimizer(s) {unknown}, must be a subset of {PINNED_OPTIMIZERS}")
        optimizers = requested
        print(f"[INFO] {size_key}: restricting to requested optimizer subset: {optimizers}")
    else:
        optimizers = PINNED_OPTIMIZERS

    runs = [
        (objective, optimizer, seed)
        for objective in objectives
        for optimizer in optimizers
        for seed in range(N_SEEDS)
    ]
    print(f"Running {len(runs)} simulations ({len(objectives)} objectives x {len(optimizers)} optimizers x {N_SEEDS} seeds)")

    all_frames = []
    t0 = time.time()
    n_failed = 0
    for i, (objective, optimizer, seed) in enumerate(runs, 1):
        try:
            all_frames.append(run_one(blackbox, objective, optimizer, seed, max_t, pinned_step, budget))
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"[WARN] run {objective}/{optimizer}/seed={seed} failed: {exc}")
        if i % 25 == 0 or i == len(runs):
            print(f"  {i}/{len(runs)} done ({time.time() - t0:.0f}s elapsed, {n_failed} failed)", flush=True)

    raw = pd.concat(all_frames, ignore_index=True)
    keep_cols = [
        "search_objective", "optimizer", "seed", "trial_id",
        "config_lr", "config_beta", FIDELITY_ATTR, "st_tuner_time",
    ] + blackbox.objectives_names
    raw = raw[keep_cols].rename(columns={"config_lr": "lr", "config_beta": "beta"})

    out_dir = paths.results_dir.parent / ("results_pinned400" if budget_tag == "generous" else f"results_pinned400_{budget_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "simulation_raw.csv"
    raw.to_csv(out_csv, index=False)
    print(f"Saved raw per-report results: {out_csv} ({len(raw)} rows)")


if __name__ == "__main__":
    main()
