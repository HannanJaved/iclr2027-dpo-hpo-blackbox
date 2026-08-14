#!/usr/bin/env python3
"""
collect_data.py

Scans the real DPO-AO training sweep for one Qwen3 model size (9 configs = 3
learning rates x 3 betas for 0.6b/1.7b/8b/14b, 16 configs = 4x4 for 4b -- the
grid shape is discovered from disk, not assumed), each with 10 intermediate
checkpoints + 1 final checkpoint, and gathers, for every (config, checkpoint)
pair:

  - the hyperparameters (lr, beta)
  - the fidelity (DPO training step)
  - all 11 benchmark scores tracked by the ICLR eval pipeline
    (qwen3_iclr_common.load_scores_for_model)
  - a cumulative elapsed-time estimate in seconds, derived from each run's own
    ``trainer_state.json`` (train_runtime / global_step gives a per-config
    seconds-per-step rate; real per-checkpoint timestamps aren't logged, so we
    linearly interpolate along that rate)

Data-quality note (4B specifically): 3 of the 16 final-checkpoint
trainer_state.json files report a train_runtime of ~2-8 seconds for 2031
steps (~0.001-0.0014 s/step) -- almost certainly a crashed-and-resumed run
whose log only captured the last restart's short segment, not the true
cumulative time. Genuine rates observed across all 5 sizes range ~1.9-34
s/step, so any rate <= RATE_SANITY_FLOOR is treated as corrupted and replaced
with the median rate of the OTHER configs of the same size (repairs 3/74
final configs across all 5 sizes; flagged below either way).

This raw grid is the source of truth the blackbox is built from (build_blackbox.py).
Kept as a separate, inspectable CSV so the (slow, filesystem-scanning) collection
step only has to run once per size.

Usage:
    python collect_data.py --size 1.7b
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import BENCHMARKS, get_paths
from qwen3_iclr_common import (
    load_dpo_final_models,
    load_dpo_intermediate_models,
    load_scores_for_model,
    model_path,
    parse_dpo_lr_beta,
    step_from_name,
)

RATE_SANITY_FLOOR = 0.5  # seconds/step; genuine rates are all > 1.8 s/step


def raw_runtime_and_step(final_model_name: str) -> tuple[float, int] | None:
    """Return (train_runtime, global_step) from the run's trainer_state.json."""
    path = model_path(final_model_name)
    if not path:
        return None
    try:
        state = json.loads(open(f"{path}/trainer_state.json").read())
    except (OSError, json.JSONDecodeError):
        return None
    global_step = state.get("global_step")
    log_history = state.get("log_history", [])
    train_runtime = next(
        (entry["train_runtime"] for entry in reversed(log_history) if "train_runtime" in entry),
        None,
    )
    if not global_step or not train_runtime:
        return None
    return train_runtime, int(global_step)


def collect(size_key: str) -> pd.DataFrame:
    final_models = load_dpo_final_models(size_key)
    intermediate_models = load_dpo_intermediate_models(size_key)
    assert final_models, f"No DPO final models found for size {size_key}"

    # Pass 1: raw (runtime, global_step) per config, to compute a robust
    # per-size fallback rate for any corrupted entries.
    raw_by_config: dict[str, tuple[float, int]] = {}
    for final_name in final_models:
        r = raw_runtime_and_step(final_name)
        if r is not None:
            raw_by_config[final_name] = r

    rates = {name: runtime / step for name, (runtime, step) in raw_by_config.items()}
    valid_rates = sorted(r for r in rates.values() if r > RATE_SANITY_FLOOR)
    fallback_rate = valid_rates[len(valid_rates) // 2] if valid_rates else None
    corrupted = [name for name, r in rates.items() if r <= RATE_SANITY_FLOOR]
    if corrupted:
        print(f"[WARN] {size_key}: corrupted train_runtime for {corrupted} "
              f"(rate <= {RATE_SANITY_FLOOR}s/step) -> using per-size median rate "
              f"{fallback_rate:.2f}s/step instead")

    rows: list[dict] = []
    for final_name in final_models:
        lr, beta = parse_dpo_lr_beta(final_name)
        if lr is None:
            print(f"[WARN] could not parse lr/beta from {final_name!r}, skipping")
            continue
        if final_name not in raw_by_config:
            print(f"[WARN] no trainer_state.json / train_runtime for {final_name!r}, skipping")
            continue

        _, final_step = raw_by_config[final_name]
        rate = rates[final_name] if rates[final_name] > RATE_SANITY_FLOOR else fallback_rate
        if rate is None:
            print(f"[WARN] no valid rate available at all for {size_key}, skipping {final_name!r}")
            continue

        step_models = sorted(
            {
                (step_from_name(name), name)
                for name in intermediate_models
                if name.startswith(final_name + "-step") and step_from_name(name) is not None
            }
        )
        checkpoints = [(step, name) for step, name in step_models]
        checkpoints.append((final_step, final_name))

        for step, model_name in checkpoints:
            scores = load_scores_for_model(model_name, judge_family="qwen")
            missing = [b for b in BENCHMARKS if scores.get(b) is None]
            if missing:
                print(f"[WARN] {model_name}: missing {missing}")
            row = {
                "lr": lr,
                "beta": beta,
                "dpo_step": step,
                "elapsed_time_sec": round(step * rate, 1),
                "model_name": model_name,
            }
            row.update({b: scores.get(b) for b in BENCHMARKS})
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["lr", "beta", "dpo_step"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    args = parser.parse_args()

    paths = get_paths(args.size)
    df = collect(args.size)
    n_configs = len(df[["lr", "beta"]].drop_duplicates())
    n_steps = df["dpo_step"].nunique()
    print(f"Collected {len(df)} rows: {n_configs} configs x up to {n_steps} fidelities")
    print(df.groupby(["lr", "beta"]).size().rename("n_checkpoints"))

    paths.grid_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.grid_csv, index=False)
    print(f"Saved: {paths.grid_csv}")


if __name__ == "__main__":
    main()
