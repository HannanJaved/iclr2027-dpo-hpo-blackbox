#!/usr/bin/env python3
"""
build_blackbox.py

Turns the raw grid collected by collect_data.py (for one model size) into a
Syne Tune ``BlackboxTabular`` and serializes it to disk (in the same on-disk
format used by the Syne Tune blackbox-repository, see
blackbox_tabular.serialize), so that run_simulations.py can ``deserialize()``
it directly without re-scanning the filesystem for eval results.

Design decisions:

- Configuration space = a SINGLE joint categorical ``cfg_id`` (sp.choice over
  an opaque integer id, one per (lr, beta) pair with a complete fidelity
  trajectory -- see drop_fidelity_incomplete_configs()), NOT two independent
  ``sp.choice(lr) x sp.choice(beta)`` dimensions. Independent per-dimension
  sampling would let a scheduler propose any (lr, beta) COMBINATION,
  including ones with no real data -- which is only safe if the grid is a
  full rectangle. Sampling jointly from the actual valid pairs instead means
  the config space can be rectangular, ragged, or anything in between
  (including sparse single-beta exploration points at a new lr) with zero
  special-casing -- every config WITH data is searchable, regardless of
  shape. The (cfg_id -> lr, beta) mapping is persisted alongside the
  blackbox (``paths.config_map_csv``) so run_simulations.py /
  run_pinned_fidelity_simulations.py can decode a run's sampled cfg_id back
  into real lr/beta values for reporting -- everything downstream of that
  decode (analyze_results.py included) sees plain "lr"/"beta" columns
  exactly as before, unaware the search space is joint.

  Two-column implementation detail: BlackboxTabular looks up a config via
  ``hyperparameters.set_index(hp_cols).loc[tuple(...)]``. With a SINGLE hp
  column, pandas' ``set_index`` silently builds a plain Index (sometimes even
  a RangeIndex) instead of a MultiIndex, and a 1-tuple ``.loc[...]`` lookup
  against that raises (verified directly against pandas, not assumed). So
  ``hyperparameters`` carries a second column, "_const", pinned to the single
  value 0 via a length-1 ``sp.choice([0])`` -- it contributes zero actual
  search dimensionality, it exists purely to keep hp_cols at >=2 columns so
  set_index builds a genuine MultiIndex and the lookup works.
- Fidelity = DPO training step (200, 400, ..., 2000, plus the final checkpoint
  at step ~2031). Every SEARCHABLE config has the complete step sequence by
  construction (drop_fidelity_incomplete_configs() excludes any that don't);
  the raw grid CSV itself may still have a config with only a final-step
  reading (training just finished, intermediate checkpoints not evaluated
  yet) -- real data, just not usable for the multi-fidelity search until the
  rest lands.
- Objectives = the 11 raw benchmark scores, three composite "Z-*" scores, and
  a synthetic ``elapsed_time_sec`` cost column (used by the simulator as the
  monotonic time attribute). The composite Z-scores are normalized *within
  each fidelity slice* (across the configs of this size, at a fixed step)
  rather than globally, so a config isn't rewarded just for having trained
  longer - it's ranked against its peers at the same point in training.
- Three kinds of grid irregularity show up in practice, handled differently:
    1. Scattered gaps (a benchmark failed for one config at one intermediate
       step): forward-/back-filled within that config's own step sequence --
       a missing win-rate at step 800 is filled from step 600 (or step 1000
       if it's the very first reading), never mixed across configs.
    2. A benchmark that failed for a config at EVERY fidelity (a full eval
       failure, nothing to fill from). If this affects only a few configs of
       this size's grid, those configs are dropped. If it affects a large
       fraction of them (e.g. 14B was once missing AlpacaEval for 5 of its 9
       DPO configs), dropping configs would leave gaps in the joint config
       space's coverage -- so instead the whole BENCHMARK is dropped for
       that size, and SEARCH_OBJECTIVES / composite Z-scores adapt
       accordingly (see usable_benchmarks()).
    3. A config missing whole ROWS, not just benchmark values within rows it
       has (e.g. only its final checkpoint evaluated so far, none of the 10
       intermediate ones) -- unfillable (fill_gaps() only fixes gaps WITHIN
       existing rows), so excluded from the searchable grid entirely until
       the rest of its checkpoints are evaluated (see
       drop_fidelity_incomplete_configs()). Still shows up correctly in
       analyze_results.py's ground-truth ranking, which only needs a
       config's final-fidelity row.

Usage:
    python build_blackbox.py --size 1.7b
    python build_blackbox.py --size 8b --family llama
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import syne_tune.config_space as sp
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular, serialize

from common import BENCHMARKS, DEFAULT_FAMILY, FAMILIES, FIDELITY_ATTR, TIME_OBJECTIVE, get_paths

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


def drop_fidelity_incomplete_configs(df: pd.DataFrame, size_key: str) -> pd.DataFrame:
    """A config missing whole ROWS (not just individual benchmark values
    within a row it has) can't be part of the multi-fidelity search: e.g. a
    config whose training just finished, evaluated so far only at its final
    checkpoint, with none of the 10 intermediate checkpoints scored yet --
    unlike a sparse-beta exploration point (see build()'s docstring), there's
    no legitimate way to fill 9 entirely-missing fidelities from one data
    point without fabricating data (fill_gaps() only fixes gaps WITHIN a
    config's existing rows, it can't invent missing rows).

    Dropped only from the copy used to build the SEARCHABLE blackbox; the
    on-disk grid CSV is untouched, so analyze_results.py's ground-truth
    ranking (which only needs each config's FINAL-fidelity row, already
    present here) still reports on it correctly.
    """
    n_rows = df.groupby(["lr", "beta"]).size()
    max_rows = n_rows.max()
    incomplete = n_rows[n_rows < max_rows]
    if len(incomplete):
        for (lr, beta), n in incomplete.items():
            print(f"[WARN] {size_key}: excluding lr={lr:.0e}, beta={beta} from the SEARCHABLE HPO grid -- "
                  f"only {n}/{max_rows} fidelities evaluated (missing checkpoints, not just missing "
                  f"benchmark values within existing ones -- can't be filled). Still fully present in the "
                  f"raw grid CSV and in analyze_results.py's ground-truth ranking (which only needs its "
                  f"final-fidelity row), just not searchable by the simulated optimizers until its "
                  f"remaining checkpoints are evaluated.")
        keep = pd.MultiIndex.from_frame(df[["lr", "beta"]]).isin(n_rows[n_rows == max_rows].index)
        df = df[keep]
    return df


def build(size_key: str, family: str = DEFAULT_FAMILY) -> tuple[BlackboxTabular, pd.DataFrame]:
    paths = get_paths(size_key, family=family)
    df = pd.read_csv(paths.grid_csv)
    df = drop_fidelity_incomplete_configs(df, size_key)
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    objectives = benchmarks + ["Z-Static", "Z-Dynamic", "Z-All", TIME_OBJECTIVE]

    fidelity_values = np.sort(df["dpo_step"].unique())
    # Every (lr, beta) pair actually present (with a complete fidelity
    # trajectory) becomes one category of the joint cfg_id search dimension
    # -- no rectangle required, sparse single-beta exploration points at a
    # new lr included right alongside everything else.
    configs = df[["lr", "beta"]].drop_duplicates().sort_values(["lr", "beta"]).reset_index(drop=True)
    configs["cfg_id"] = configs.index

    n_evals, n_seeds, n_fidelities, n_objectives = (
        len(configs), 1, len(fidelity_values), len(objectives),
    )
    objectives_evaluations = np.full((n_evals, n_seeds, n_fidelities, n_objectives), np.nan)

    fidelity_index = {step: i for i, step in enumerate(fidelity_values)}
    # .iterrows() upcasts a mixed-dtype row (float lr/beta + int cfg_id) to a
    # common dtype, silently turning cfg_id into a float -- itertuples() keeps
    # each column's own dtype instead.
    config_index = {(row.lr, row.beta): int(row.cfg_id) for row in configs.itertuples()}

    for _, row in df.iterrows():
        ci = config_index[(row["lr"], row["beta"])]
        fi = fidelity_index[row["dpo_step"]]
        objectives_evaluations[ci, 0, fi, :] = [row[obj] for obj in objectives]

    assert not np.isnan(objectives_evaluations).any(), "every (config, fidelity) cell must be filled"
    assert len(configs) == len(configs.drop_duplicates(["lr", "beta"])), "duplicate (lr, beta) rows in the grid"

    # BlackboxTabular's lookup does hyperparameters.set_index(hp_cols) and then
    # .loc[tuple(...)] -- with a SINGLE hp column, pandas' set_index silently
    # builds a plain Index (or even a RangeIndex) instead of a MultiIndex, and
    # a 1-tuple .loc lookup against that raises (verified directly against
    # pandas, not assumed: `zip() argument 2 is longer than argument 1`). Two
    # or more hp columns make set_index build a genuine MultiIndex, where the
    # same lookup works correctly. "_const" is a second column that always
    # takes the single value 0 -- it carries zero information (configured as
    # a length-1 sp.choice(), so every sample is forced to 0) and exists
    # purely to keep hyperparameters at >=2 columns and dodge this pandas
    # single-level-Index quirk.
    configs["_const"] = 0
    configuration_space = {"cfg_id": sp.choice(configs["cfg_id"].tolist()), "_const": sp.choice([0])}
    fidelity_space = {FIDELITY_ATTR: sp.randint(0, int(fidelity_values.max()))}

    n_betas_by_lr = configs.groupby("lr")["beta"].nunique()
    sparse = n_betas_by_lr[n_betas_by_lr < n_betas_by_lr.max()]
    if len(sparse):
        print(f"[INFO] {size_key}: {len(sparse)} lr value(s) have fewer betas than the rest "
              f"({dict(sparse)} vs up to {n_betas_by_lr.max()}) -- included in the search via the joint "
              f"cfg_id categorical (every real config is independently searchable; no rectangle required).")

    blackbox = BlackboxTabular(
        hyperparameters=configs[["cfg_id", "_const"]],
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objectives_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objectives,
    )
    return blackbox, configs[["cfg_id", "lr", "beta"]]


def sanity_check(blackbox: BlackboxTabular, config_map: pd.DataFrame) -> None:
    print(blackbox)
    cfg = blackbox.hyperparameters.iloc[0].to_dict()
    lr, beta = config_map.loc[config_map["cfg_id"] == cfg["cfg_id"], ["lr", "beta"]].iloc[0]
    result = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[0])})
    print(f"Spot check {cfg} (lr={lr}, beta={beta}) @ step {blackbox.fidelity_values[0]}: {result}")
    result_final = blackbox.objective_function(cfg, fidelity={FIDELITY_ATTR: int(blackbox.fidelity_values[-1])})
    print(f"Spot check {cfg} @ step {blackbox.fidelity_values[-1]}: {result_final}")
    time_idx = blackbox.objectives_names.index(TIME_OBJECTIVE)
    times = blackbox.objectives_evaluations[0, 0, :, time_idx]
    assert np.all(np.diff(times) > 0), "elapsed_time_sec must be strictly increasing per config"
    print("elapsed_time_sec is monotonically increasing: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    args = parser.parse_args()

    paths = get_paths(args.size, family=args.family)
    blackbox, config_map = build(args.size, args.family)
    sanity_check(blackbox, config_map)

    paths.blackbox_dir.parent.mkdir(parents=True, exist_ok=True)
    serialize(
        {paths.blackbox_key: blackbox},
        path=str(paths.blackbox_dir),
        metadata={"size_key": args.size, "family": args.family, "source": f"ICLR27 {args.family} DPO-AO sweep"},
    )
    config_map.to_csv(paths.config_map_csv, index=False)
    print(f"Serialized blackbox to: {paths.blackbox_dir}")
    print(f"Saved cfg_id -> (lr, beta) map to: {paths.config_map_csv}")


if __name__ == "__main__":
    main()
