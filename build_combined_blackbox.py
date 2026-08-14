#!/usr/bin/env python3
"""
build_combined_blackbox.py

Builds ONE joint blackbox spanning all 5 Qwen3 sizes, where "size" is itself
a searchable hyperparameter alongside lr and beta -- a genuinely different
experiment from the rest of this study (which always fixed size and searched
only within it). This answers: if an optimizer could pick model size *and*
DPO hyperparameters together under one compute budget, what would it choose,
and does it correctly trade off a 14B run's ~6.7x higher cost against its
(usually, but not always) higher achievable quality?

Design (see the AskUserQuestion decisions this was built from):
  - Config space = size (5 choices) x lr x beta, restricted to the 3x3 lr/beta
    subgrid shared by every size (lr in {1e-6,2e-6,4e-6}, beta in
    {0.01,0.02,0.04}). 4B's wider 4x4 sweep is subset down to this shared
    grid so the combined config space stays a clean rectangle: 45 configs
    total (5 sizes x 9 lr/beta combos), reusing the per-size grid CSVs
    collect_data.py already built (no new data collection needed).
  - Objectives: ELO (already computed from one shared tournament pool across
    all sizes, so it's directly comparable in absolute terms) and Z-All
    recomputed by pooling ALL 45 configs together at each fidelity (see
    add_composite_zscores in build_blackbox.py, which normalizes within
    whatever rows share a dpo_step -- passing it the full 45-config frame
    instead of one size's 9 gives a genuinely cross-size-normalized score,
    unlike the per-size Z-All used everywhere else in this study).
  - Missing evals (AlpacaEval fully missing for 11% of the 45 configs, Arena-
    Hard for 2%) are imputed from the mean of OTHER CONFIGS OF THE SAME SIZE
    at that step (peer_cols=["size"]), not pooled across sizes -- a missing
    14B cell shouldn't be filled in with mostly-0.6B/1.7B numbers.
  - elapsed_time_sec is untouched per-config real cost (0.6B ~10.3k s/run up
    to 14B ~68.7k s/run) -- this cost spread is the whole point of the
    experiment, so it's deliberately NOT normalized away.

Usage:
    python build_combined_blackbox.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize

from build_blackbox import add_composite_zscores, fill_gaps, sanity_check, usable_benchmarks
from common import (
    COMBINED_BETA_VALUES,
    COMBINED_LR_VALUES,
    FIDELITY_ATTR,
    SIZE_KEYS,
    get_combined_paths,
    get_paths,
)

GROUP_COLS = ["size", "lr", "beta"]


def load_combined_grid() -> pd.DataFrame:
    frames = []
    for size_key in SIZE_KEYS:
        df = pd.read_csv(get_paths(size_key).grid_csv)
        df = df[df["lr"].isin(COMBINED_LR_VALUES) & df["beta"].isin(COMBINED_BETA_VALUES)].copy()
        df["size"] = size_key
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    n_configs = combined[GROUP_COLS].drop_duplicates().shape[0]
    assert n_configs == len(SIZE_KEYS) * len(COMBINED_LR_VALUES) * len(COMBINED_BETA_VALUES), \
        f"expected a full {len(SIZE_KEYS)}x{len(COMBINED_LR_VALUES)}x{len(COMBINED_BETA_VALUES)} rectangle, got {n_configs} configs"
    return combined


def build() -> BlackboxTabular:
    df = load_combined_grid()
    benchmarks = usable_benchmarks(df, "combined", group_cols=GROUP_COLS)
    df = fill_gaps(df, benchmarks, "combined", group_cols=GROUP_COLS, peer_cols=["size"])
    df = add_composite_zscores(df, benchmarks)  # pooled across all 45 configs -> cross-size normalized
    objectives = benchmarks + ["Z-Static", "Z-Dynamic", "Z-All", "elapsed_time_sec"]

    fidelity_values = np.sort(df["dpo_step"].unique())
    configs = df[GROUP_COLS].drop_duplicates().sort_values(GROUP_COLS).reset_index(drop=True)

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    config_index = {(row["size"], row["lr"], row["beta"]): i for i, row in configs.iterrows()}

    for _, row in df.iterrows():
        ci = config_index[(row["size"], row["lr"], row["beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"

    configuration_space = {
        "size": sp.choice(SIZE_KEYS),
        "lr": sp.choice(COMBINED_LR_VALUES),
        "beta": sp.choice(COMBINED_BETA_VALUES),
    }
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}

    blackbox = BlackboxTabular(
        hyperparameters=configs[GROUP_COLS],
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )
    return blackbox


def main() -> None:
    paths = get_combined_paths()
    blackbox = build()
    sanity_check(blackbox)

    paths.grid_csv.parent.mkdir(parents=True, exist_ok=True)
    load_combined_grid().to_csv(paths.grid_csv, index=False)
    print(f"Saved combined grid: {paths.grid_csv}")

    paths.blackbox_dir.parent.mkdir(parents=True, exist_ok=True)
    serialize(
        {"qwen3-combined-dpo-ao": blackbox},
        path=str(paths.blackbox_dir),
        metadata={"source": "ICLR27 qwen3 DPO-AO sweep, combined across sizes"},
    )
    print(f"Serialized blackbox to: {paths.blackbox_dir}")


if __name__ == "__main__":
    main()
