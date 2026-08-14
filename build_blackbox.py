#!/usr/bin/env python3
"""
build_blackbox.py

Turns the raw grid collected by collect_data.py (for one model size) into a
Syne Tune ``BlackboxTabular`` and serializes it to disk (in the same on-disk
format used by the Syne Tune blackbox-repository, see
blackbox_tabular.serialize), so that run_simulations.py can ``deserialize()``
it directly without re-scanning the filesystem for eval results.

Design decisions:

- Configuration space = {lr, beta}, both represented as ``sp.choice`` over the
  exact values swept for this size (categorical/finite) -- 3x3 for
  0.6b/1.7b/8b/14b, 4x4 for 4b. Every sampled configuration is therefore an
  exact hit in the table; no surrogate / interpolation needed.
- Fidelity = DPO training step (200, 400, ..., 2000, plus the final checkpoint
  at step ~2031). Same steps are available for every config of a given size.
- Objectives = the 11 raw benchmark scores, three composite "Z-*" scores, and
  a synthetic ``elapsed_time_sec`` cost column (used by the simulator as the
  monotonic time attribute). The composite Z-scores are normalized *within
  each fidelity slice* (across the configs of this size, at a fixed step)
  rather than globally, so a config isn't rewarded just for having trained
  longer - it's ranked against its peers at the same point in training.
- Two kinds of eval gaps show up in practice, handled differently:
    1. Scattered gaps (a benchmark failed for one config at one intermediate
       step): forward-/back-filled within that config's own step sequence --
       a missing win-rate at step 800 is filled from step 600 (or step 1000
       if it's the very first reading), never mixed across configs.
    2. A benchmark that failed for a config at EVERY fidelity (a full eval
       failure, nothing to fill from). If this affects only a few configs of
       this size's grid, those configs are dropped. If it affects a large
       fraction of them (e.g. 14B is missing AlpacaEval for 5 of its 9 DPO
       configs), dropping configs would leave a ragged, non-rectangular
       config space -- so instead the whole BENCHMARK is dropped for that
       size, and SEARCH_OBJECTIVES / composite Z-scores adapt accordingly
       (see usable_benchmarks()).

Usage:
    python build_blackbox.py --size 1.7b
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize

from common import BENCHMARKS, FIDELITY_ATTR, TIME_OBJECTIVE, get_paths

STATIC_BENCHMARKS = ["ARC-C", "GPQA", "GSM8K", "HellaSwag", "PIQA", "TruthfulQA", "IFEval"]
DYNAMIC_BENCHMARKS = ["Arena-Hard", "MT-Bench", "AlpacaEval", "ELO"]

# If a benchmark is fully missing (no data at any fidelity) for more than this
# fraction of a size's configs, drop the benchmark for that size instead of
# dropping that many configs (which would break the rectangular config space).
DROP_BENCHMARK_THRESHOLD = 0.2


def usable_benchmarks(df: pd.DataFrame, size_key: str, group_cols: list[str] | None = None) -> list[str]:
    """``group_cols`` identifies a "config" for the purpose of counting
    missingness -- defaults to (lr, beta) for a single size's grid; pass
    (size, lr, beta) when building the combined cross-size blackbox."""
    if group_cols is None:
        group_cols = ["lr", "beta"]
    n_configs = df[group_cols].drop_duplicates().shape[0]
    fully_missing_frac = {
        b: df.groupby(group_cols)[b].apply(lambda s: s.isna().all()).mean()
        for b in BENCHMARKS
    }
    dropped = [b for b, frac in fully_missing_frac.items() if frac > DROP_BENCHMARK_THRESHOLD]
    for b in dropped:
        print(f"[WARN] {size_key}: dropping benchmark {b!r} entirely "
              f"(missing for {fully_missing_frac[b]:.0%} of {n_configs} configs)")
    return [b for b in BENCHMARKS if b not in dropped]


def fill_gaps(df: pd.DataFrame, benchmarks: list[str], size_key: str,
              group_cols: list[str] | None = None, peer_cols: list[str] | None = None) -> pd.DataFrame:
    """Forward-/back-fill scattered missing evals within each config's own
    step sequence. usable_benchmarks() already dropped any benchmark that was
    a systemic problem (missing for a large fraction of configs), so any
    per-config gap still standing after ffill/bfill is by construction a rare,
    isolated straggler -- dropping its config would leave a hole in the
    config space that a plain per-dimension sp.choice() can't represent
    safely (it could sample the missing combo). So instead of dropping, that
    one cell is imputed from the per-fidelity mean across PEER configs -- a
    disclosed, narrowly-scoped exception to the "never mixed across configs"
    rule, used only for rare isolated holes.

    :param group_cols: identifies a "config" -- defaults to (lr, beta); pass
        (size, lr, beta) for the combined cross-size blackbox.
    :param peer_cols: which other configs count as "peers" to impute from,
        via a per-(dpo_step, *peer_cols) mean. Defaults to every other config
        of this size. For the combined blackbox, pass ["size"] so a missing
        14B cell is imputed from OTHER 14B configs, not diluted by smaller
        (and structurally different) sizes.
    """
    if group_cols is None:
        group_cols = ["lr", "beta"]
    if peer_cols is None:
        peer_cols = []
    df = df.sort_values(group_cols + ["dpo_step"]).copy()
    df[benchmarks] = df.groupby(group_cols)[benchmarks].transform(
        lambda col: col.ffill().bfill()
    )
    still_missing = df.groupby(group_cols)[benchmarks].apply(lambda g: g.isna().any())
    bad_configs = still_missing[still_missing.any(axis=1)]
    for key, row in bad_configs.iterrows():
        key_tuple = key if isinstance(key, tuple) else (key,)
        missing = row[row].index.tolist()
        key_desc = ", ".join(f"{c}={v}" for c, v in zip(group_cols, key_tuple))
        print(f"[WARN] {size_key}: config {key_desc} has no data at ANY fidelity for "
              f"{missing} -> imputing from the per-step mean of {'peer' if peer_cols else 'other'} configs")
        mask = np.logical_and.reduce([df[c] == v for c, v in zip(group_cols, key_tuple)])
        for col in missing:
            peer_mean = df.groupby(peer_cols + ["dpo_step"])[col].transform("mean") if peer_cols \
                else df.groupby("dpo_step")[col].transform("mean")
            df.loc[mask, col] = df.loc[mask, col].fillna(peer_mean[mask])
    assert not df[benchmarks].isna().any().any(), "unfillable gaps remain"
    return df


def add_composite_zscores(df: pd.DataFrame, benchmarks: list[str] | None = None) -> pd.DataFrame:
    """Per-fidelity (across the configs of this size at a fixed step),
    z-normalize each benchmark and average within static/dynamic/all groups.
    ``benchmarks`` defaults to all 11 (only pass a subset if some were
    dropped for this size, see usable_benchmarks())."""
    if benchmarks is None:
        benchmarks = BENCHMARKS
    static = [b for b in STATIC_BENCHMARKS if b in benchmarks]
    dynamic = [b for b in DYNAMIC_BENCHMARKS if b in benchmarks]
    df = df.copy()
    z = df.groupby("dpo_step")[benchmarks].transform(
        lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) > 1e-12 else 0.0
    )
    df["Z-Static"] = z[static].mean(axis=1) if static else np.nan
    df["Z-Dynamic"] = z[dynamic].mean(axis=1) if dynamic else np.nan
    df["Z-All"] = z[benchmarks].mean(axis=1)
    return df


def build(size_key: str) -> BlackboxTabular:
    paths = get_paths(size_key)
    df = pd.read_csv(paths.grid_csv)
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    objectives = benchmarks + ["Z-Static", "Z-Dynamic", "Z-All", TIME_OBJECTIVE]

    fidelity_values = np.sort(df["dpo_step"].unique())
    configs = df[["lr", "beta"]].drop_duplicates().sort_values(["lr", "beta"]).reset_index(drop=True)
    lr_values = sorted(configs["lr"].unique().tolist())
    beta_values = sorted(configs["beta"].unique().tolist())

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    config_index = {(row["lr"], row["beta"]): i for i, row in configs.iterrows()}

    for _, row in df.iterrows():
        ci = config_index[(row["lr"], row["beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"
    assert len(configs) == len(configs.drop_duplicates(["lr", "beta"])), "config grid must stay rectangular"
    n_expected = len(lr_values) * len(beta_values)
    assert len(configs) == n_expected, (
        f"{size_key}: config grid is not a full lr x beta rectangle after dropping "
        f"({len(configs)} configs, expected {n_expected} = {len(lr_values)} lrs x {len(beta_values)} betas) "
        f"-- categorical sp.choice(lr) x sp.choice(beta) would allow invalid combos"
    )

    configuration_space = {
        "lr": sp.choice(lr_values),
        "beta": sp.choice(beta_values),
    }
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}

    blackbox = BlackboxTabular(
        hyperparameters=configs[["lr", "beta"]],
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )
    return blackbox


def sanity_check(blackbox: BlackboxTabular) -> None:
    print(blackbox)
    cfg = blackbox.hyperparameters.iloc[0].to_dict()
    result = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[0])})
    print(f"Spot check {cfg} @ step {blackbox.fidelity_values[0]}: {result}")
    result_final = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[-1])})
    print(f"Spot check {cfg} @ step {blackbox.fidelity_values[-1]}: {result_final}")
    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    times = blackbox.objectives_evaluations[0, 0, :, time_idx]
    assert np.all(np.diff(times) > 0), "elapsed_time_sec must be strictly increasing per config"
    print("elapsed_time_sec is monotonically increasing: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    args = parser.parse_args()

    paths = get_paths(args.size)
    blackbox = build(args.size)
    sanity_check(blackbox)

    paths.blackbox_dir.parent.mkdir(parents=True, exist_ok=True)
    serialize(
        {f"qwen3-{args.size}-dpo-ao": blackbox},
        path=str(paths.blackbox_dir),
        metadata={"size_key": args.size, "source": "ICLR27 qwen3 DPO-AO sweep"},
    )
    print(f"Serialized blackbox to: {paths.blackbox_dir}")


if __name__ == "__main__":
    main()
