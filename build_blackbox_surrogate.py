#!/usr/bin/env python3
"""
build_blackbox_surrogate.py

Continuous-space counterpart to build_blackbox.py. The per-size blackbox that
script produces is purely tabular -- its configuration space is a single
joint categorical (cfg_id) over the exact (lr, beta) pairs that were actually
measured, because BlackboxTabular can only answer queries at points it has
real data for (see build_blackbox.py's docstring). A continuous or
integer-valued scheduler can't search that space: any off-grid point it
proposes has nothing to look up.

This script instead fits a regression surrogate (scikit-learn API, via
syne_tune.blackbox_repository's add_surrogate) on top of the SAME cleaned
grid data build_blackbox.py uses (same usable_benchmarks/fill_gaps/
add_composite_zscores pipeline, reused directly from that module), then wraps
it in a genuinely continuous configuration space bounded to the observed
grid's own min/max on each dimension (extrapolating outside the range a
surrogate was fit on isn't defensible, so the search space doesn't go
there). Fidelity stays the real discrete dpo_step values -- only lr/beta
become continuous.

--surrogate picks the regression model:
  knn1  KNeighborsRegressor(n_neighbors=1) -- syne-tune's own default. Exact
        at every real grid point, but LOCALLY CONSTANT everywhere else: the
        continuous space collapses into as many flat Voronoi cells as there
        are real grid points (9-16/size here), and a model-based optimizer
        (TPE, CQR) has no smooth local structure to exploit -- empirically,
        this made TPE perform *worse* than RandomSearch (see the run history
        for this file). Kept as a choice for comparison, not the default.
  knn3  KNeighborsRegressor(n_neighbors=3, weights="distance") -- same idea
        as knn5 with a tighter neighborhood (less smoothing, closer to knn1).
  knn5  KNeighborsRegressor(n_neighbors=5, weights="distance") -- cheap fix:
        blends the 5 nearest real points instead of snapping to 1, so the
        surface actually varies within a neighborhood instead of being flat.
  gp    GaussianProcessRegressor (Matern(nu=2.5) + WhiteKernel, normalize_y)
        -- genuinely smooth interpolation with calibrated uncertainty, the
        standard choice for small-N (9-16 points/size here) continuous
        surrogate modeling.

lr/beta are fit and searched in LOG10 SPACE (columns "log_lr"/"log_beta"),
not raw. This matters beyond just how schedulers sample: add_surrogate's
feature pipeline (blackbox_surrogate.py:195-240) only ever sees the columns
we hand it and applies a plain StandardScaler to them -- it has no notion of
the config-space domain being "loguniform", so a raw-space surrogate (knn5
or gp) would compute "nearest"/kernel distance on raw lr, badly distorting
locality on a grid built from doubling steps (1e-6, 2e-6, 4e-6, 8e-6, ...).
Pre-transforming to log10 makes the surrogate's own distance metric scale-
aware, not just the scheduler's sampling. knn1 doesn't need this as urgently
(nearest-neighbor RANKING is fairly robust to it) but uses the same log
features for a consistent, comparable 3-way setup.

Output is NOT syne-tune's tabular serialize() format (that's array-based,
unsuited to a fitted sklearn pipeline) -- it's a plain pickle of the fitted
BlackboxSurrogate object, at
<size_dir>/blackbox_surrogate/<family>-<size>-dpo-ao-surrogate-<surrogate>.pkl.

Usage:
    python build_blackbox_surrogate.py --size 4b --surrogate gp
    python build_blackbox_surrogate.py --size 8b --surrogate knn5 --family llama
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository import add_surrogate
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular

from build_blackbox import add_composite_zscores, drop_fidelity_incomplete_configs, fill_gaps, usable_benchmarks
from common import DEFAULT_FAMILY, FAMILIES, FIDELITY_ATTR, TIME_OBJECTIVE, get_paths

SURROGATE_CHOICES = ["knn1", "knn3", "knn5", "gp"]
DEFAULT_SURROGATE = "gp"


def make_surrogate_model(name: str):
    if name == "knn1":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=1)
    if name == "knn3":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=3, weights="distance")
    if name == "knn5":
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=5, weights="distance")
    if name == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-2)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    raise ValueError(name)


def surrogate_path(size_key: str, family: str, surrogate: str):
    size_dir = get_paths(size_key, family=family).size_dir
    out_dir = size_dir / "blackbox_surrogate"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{family}-{size_key}-dpo-ao-surrogate-{surrogate}.pkl"


def get_surrogate_results_paths(size_key: str, family: str, surrogate: str = DEFAULT_SURROGATE):
    """Mirrors common.get_paths()'s results-dir naming, under a parallel
    results_surrogate_<surrogate> tree per surrogate model, so the variants
    (and build_blackbox.py's discrete-grid results) never collide."""
    size_dir = get_paths(size_key, family=family).size_dir
    results_dirname = f"results_surrogate_{surrogate}"
    results_dir = size_dir / results_dirname
    return {
        "results_dir": results_dir,
        "simulation_raw_csv": results_dir / "simulation_raw.csv",
        "best_found_csv": results_dir / "best_found.csv",
        "figures_dir": results_dir / "figures",
    }


def build(size_key: str, family: str = DEFAULT_FAMILY, surrogate: str = DEFAULT_SURROGATE):
    """Same cleaned-grid pipeline as build_blackbox.build(), but keeps real
    lr/beta as log10-transformed float columns instead of collapsing them
    into an opaque cfg_id -- a continuous surrogate needs literal numeric
    features to regress on, and log10 so the surrogate's own distance/kernel
    computations are scale-aware (see module docstring)."""
    paths = get_paths(size_key, family=family)
    df = pd.read_csv(paths.grid_csv)
    df = drop_fidelity_incomplete_configs(df, size_key)
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    objectives = benchmarks + ["Z-Static", "Z-Dynamic", "Z-All", TIME_OBJECTIVE]

    df["log_lr"] = np.log10(df["lr"])
    df["log_beta"] = np.log10(df["beta"])

    fidelity_values = np.sort(df["dpo_step"].unique())
    configs = df[["log_lr", "log_beta"]].drop_duplicates().sort_values(["log_lr", "log_beta"]).reset_index(drop=True)

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    config_index = {(row.log_lr, row.log_beta): i for i, row in enumerate(configs.itertuples())}

    for _, row in df.iterrows():
        ci = config_index[(row["log_lr"], row["log_beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"

    # Discrete placeholder blackbox, never serialized or searched directly --
    # exists only so add_surrogate() can extract X/y from it via the same
    # tested hyperparameter_objectives_values() reshaping logic BlackboxTabular
    # already implements, instead of hand-rolling that reshape here.
    discrete_configuration_space = {
        "log_lr": sp.choice(sorted(configs["log_lr"].unique().tolist())),
        "log_beta": sp.choice(sorted(configs["log_beta"].unique().tolist())),
    }
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}
    placeholder = BlackboxTabular(
        hyperparameters=configs[["log_lr", "log_beta"]],
        configuration_space=discrete_configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )

    log_lr_min, log_lr_max = float(configs["log_lr"].min()), float(configs["log_lr"].max())
    log_beta_min, log_beta_max = float(configs["log_beta"].min()), float(configs["log_beta"].max())
    continuous_configuration_space = {
        "log_lr": sp.uniform(log_lr_min, log_lr_max),  # uniform in log10-space == log-uniform in real space
        "log_beta": sp.uniform(log_beta_min, log_beta_max),
    }

    surrogate_blackbox = add_surrogate(
        placeholder,
        surrogate=make_surrogate_model(surrogate),
        configuration_space=continuous_configuration_space,
        predict_curves=True,
    )
    return surrogate_blackbox, (log_lr_min, log_lr_max), (log_beta_min, log_beta_max)


def sanity_check(blackbox, log_lr_range: tuple[float, float], log_beta_range: tuple[float, float]) -> None:
    print(blackbox)
    fidelity_values = blackbox.fidelity_values
    log_lr_mid = (log_lr_range[0] + log_lr_range[1]) / 2  # midpoint in log-space -- an OFF-GRID query
    log_beta_mid = (log_beta_range[0] + log_beta_range[1]) / 2
    cfg = {"log_lr": log_lr_mid, "log_beta": log_beta_mid}

    # NOTE: BlackboxSurrogate._objective_function has a real bug (in vendored
    # syne-tune, not our code) when predict_curves=True and a single fidelity
    # is requested directly -- it indexes the (num_fidelities, num_objectives)
    # prediction array with a length-1 ARRAY instead of a scalar, which keeps
    # an extra dimension and makes dict(zip(objectives_names, prediction[ind]))
    # zip against one row instead of one scalar per objective (silently wrong,
    # not an exception). Work around it: query with fidelity=None to get the
    # full (num_fidelities, num_objectives) curve back as a plain array
    # (that code path doesn't have the bug), then index the fidelity we want
    # ourselves. Anything else built on top of this blackbox (run_simulations
    # -style consumers included) needs the same workaround.
    curve = blackbox.objective_function(cfg)  # (num_fidelities, num_objectives)
    for step in (fidelity_values[0], fidelity_values[-1]):
        fi = int(np.where(fidelity_values == step)[0][0])
        result = dict(zip(blackbox.objectives_names, curve[fi]))
        print(f"Off-grid spot check lr={10**log_lr_mid:.2e} beta={10**log_beta_mid:.4f} @ step {step}: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    parser.add_argument("--surrogate", default=DEFAULT_SURROGATE, choices=SURROGATE_CHOICES)
    args = parser.parse_args()

    blackbox, log_lr_range, log_beta_range = build(args.size, family=args.family, surrogate=args.surrogate)
    sanity_check(blackbox, log_lr_range, log_beta_range)

    out_path = surrogate_path(args.size, args.family, args.surrogate)
    with open(out_path, "wb") as fh:
        pickle.dump(blackbox, fh)
    print(f"Saved {args.surrogate} surrogate blackbox to: {out_path}")
    print(f"Continuous search space: lr in [{10**log_lr_range[0]:.2e}, {10**log_lr_range[1]:.2e}], "
          f"beta in [{10**log_beta_range[0]:.4f}, {10**log_beta_range[1]:.4f}] (both log-uniform)")


if __name__ == "__main__":
    main()
