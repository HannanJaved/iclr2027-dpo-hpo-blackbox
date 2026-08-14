#!/usr/bin/env python3
"""
analyze_transfer_across_fidelity.py

The same "cheap proxy -> apply verbatim" question as
analyze_transfer_across_scale.py, but along the training-fidelity axis
instead of the model-size axis: instead of tuning on a smaller MODEL, tune on
an EARLY CHECKPOINT of the SAME model/run (e.g. step 400 = ~20% of the 2031-
step DPO run, present at every size), then commit to that argmax config for
the rest of training. Zero new training needed -- every size already logged
all 9 shared-grid configs at every checkpoint (200, 400, ..., 2031), so this
is again a lookup: pick the argmax (lr, beta) at fidelity step_s, read off
its true quality at the final step.

Cost model (this is what makes it different from the size-transfer script):
this is early stopping WITHIN one run, so continuing the winning config from
step_s to the final step is NOT a fresh full-cost run -- it's the marginal
cost of finishing a run already partway done. So:
  cost(early-stop strategy) = 9 x cost(step_s) + 1 x (cost(final) - cost(step_s))
                            = 8 x cost(step_s) + cost(final)
  cost(native strategy)     = 9 x cost(final)
using each size's own measured elapsed_time_sec at that step (not a linear
approximation).

Produces (under figure/analyze_transfer_across_fidelity_<objective>/):
  - regret_vs_step_fraction.png : normalized regret at the final step vs.
    fraction of training used for the proxy signal (step_s / final_step),
    one line per size -- the headline "how cheap can the signal be" plot.
    Step 400 (~20%) is marked explicitly.
  - fidelity_cost_benefit.png   : % compute saved vs. regret incurred, all
    (size, step_s) points.
  - fidelity_transfer_table.csv : the raw numbers, including a step==400 slice.

Usage:
    python analyze_transfer_across_fidelity.py [--objective Z-All]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_blackbox import add_composite_zscores, fill_gaps, usable_benchmarks
from common import COMBINED_BETA_VALUES, COMBINED_LR_VALUES, HERE, SIZE_KEYS, get_paths

TRANSFER_OBJECTIVES = ["Z-All", "Z-Dynamic", "Z-Static"]
EARLY_STEP_HIGHLIGHT = 400  # ~20% of the 2031-step DPO run, present at every size


def out_dir_for(objective: str) -> Path:
    slug = objective.lower().replace("-", "_")
    d = HERE / "figure" / f"analyze_transfer_across_fidelity_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def shared_grid_all_steps(size_key: str) -> pd.DataFrame:
    """Full step-by-step grid for one size, restricted to the shared 3x3
    (lr, beta) subgrid, with composite z-scores computed per-step (matching
    build_blackbox.add_composite_zscores exactly)."""
    paths = get_paths(size_key)
    df = pd.read_csv(paths.grid_csv)
    mask = df["lr"].isin(COMBINED_LR_VALUES) & df["beta"].isin(COMBINED_BETA_VALUES)
    df = df[mask].copy()
    benchmarks = usable_benchmarks(df, size_key)
    df = fill_gaps(df, benchmarks, size_key)
    df = add_composite_zscores(df, benchmarks)
    for objective in TRANSFER_OBJECTIVES:
        df[f"rank_{objective}"] = df.groupby("dpo_step")[objective].rank(ascending=False, method="min").astype(int)
    return df.reset_index(drop=True)


def build_fidelity_table(objective: str) -> pd.DataFrame:
    n_shared = len(COMBINED_LR_VALUES) * len(COMBINED_BETA_VALUES)
    rank_col = f"rank_{objective}"
    rows = []
    for size_key in SIZE_KEYS:
        df = shared_grid_all_steps(size_key)
        final_step = df["dpo_step"].max()
        final = df[df["dpo_step"] == final_step]
        final_best_z = final[objective].max()
        final_worst_z = final[objective].min()
        span = final_best_z - final_worst_z
        span = span if span > 1e-12 else np.nan
        final_cost = float(final["elapsed_time_sec"].mean())

        for step_s in sorted(df["dpo_step"].unique()):
            if step_s == final_step:
                continue
            proxy = df[df["dpo_step"] == step_s]
            proxy_best = proxy.loc[proxy[objective].idxmax()]
            hit = final[(final["lr"] == proxy_best["lr"]) & (final["beta"] == proxy_best["beta"])]
            assert len(hit) == 1
            transferred_z = float(hit[objective].iloc[0])
            transferred_rank = int(hit[rank_col].iloc[0])
            regret = final_best_z - transferred_z
            norm_regret = regret / span if span == span else 0.0
            random_norm_regret = (final_best_z - final[objective].mean()) / span if span == span else 0.0

            proxy_cost = float(proxy["elapsed_time_sec"].mean())
            early_stop_cost = n_shared * proxy_cost + 1 * (final_cost - proxy_cost)
            native_cost = n_shared * final_cost
            pct_compute_saved = 100.0 * (1.0 - early_stop_cost / native_cost)

            rows.append({
                "size": size_key, "step_s": int(step_s), "final_step": int(final_step),
                "step_fraction": step_s / final_step,
                "transferred_lr": proxy_best["lr"], "transferred_beta": proxy_best["beta"],
                "transferred_rank_of_9": transferred_rank,
                "norm_regret": norm_regret, "random_norm_regret": random_norm_regret,
                "pct_compute_saved": pct_compute_saved,
            })
    return pd.DataFrame(rows)


def plot_regret_vs_step_fraction(table: pd.DataFrame, objective: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SIZE_KEYS)))
    for color, size_key in zip(colors, SIZE_KEYS):
        sub = table[table["size"] == size_key].sort_values("step_fraction")
        ax.plot(sub["step_fraction"], sub["norm_regret"], marker="o", color=color, label=f"Qwen3-{size_key}")
    highlight = table[table["step_s"] == EARLY_STEP_HIGHLIGHT]
    if not highlight.empty:
        frac = highlight["step_fraction"].iloc[0]
        ax.axvline(frac, color="grey", linestyle="--", linewidth=1,
                   label=f"step {EARLY_STEP_HIGHLIGHT} (~{frac:.0%} of training)")
    ax.set_xlabel("fraction of DPO training used for the proxy signal (step_s / final_step)")
    ax.set_ylabel(f"normalized {objective} regret at final step")
    ax.set_title(f"How cheap can the early-stopping signal be?\n(commit to the argmax config at step_s, objective: {objective})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "regret_vs_step_fraction.png", dpi=150)
    plt.close(fig)


def plot_cost_benefit(table: pd.DataFrame, objective: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(table["pct_compute_saved"], table["norm_regret"], c=table["step_fraction"],
                     cmap="viridis_r", s=45, alpha=0.85)
    fig.colorbar(sc, ax=ax, label="step fraction used for the proxy")
    highlight = table[table["step_s"] == EARLY_STEP_HIGHLIGHT]
    ax.scatter(highlight["pct_compute_saved"], highlight["norm_regret"], facecolors="none",
               edgecolors="red", s=140, linewidths=1.5, label=f"step {EARLY_STEP_HIGHLIGHT}")
    ax.set_xlabel("% compute saved vs. training all 9 configs to completion")
    ax.set_ylabel(f"normalized {objective} regret at final step")
    ax.set_title("Cheapest tuning strategy at fixed budget: early stopping within one size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fidelity_cost_benefit.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="Z-All", choices=TRANSFER_OBJECTIVES)
    args = parser.parse_args()
    objective = args.objective
    out_dir = out_dir_for(objective)

    table = build_fidelity_table(objective)
    table.to_csv(out_dir / "fidelity_transfer_table.csv", index=False)
    plot_regret_vs_step_fraction(table, objective, out_dir)
    plot_cost_benefit(table, objective, out_dir)

    print(f"[fidelity:{objective}] wrote {len(table)}-row table to {out_dir / 'fidelity_transfer_table.csv'}")
    step400 = table[table["step_s"] == EARLY_STEP_HIGHLIGHT]
    print(f"[fidelity:{objective}] step {EARLY_STEP_HIGHLIGHT} (~20%) results by size:")
    print(step400[["size", "step_fraction", "transferred_rank_of_9", "norm_regret",
                    "random_norm_regret", "pct_compute_saved"]].to_string(index=False))


if __name__ == "__main__":
    main()
