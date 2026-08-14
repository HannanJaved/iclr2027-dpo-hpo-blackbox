#!/usr/bin/env python3
"""
analyze_transfer_across_scale.py

Answers the "Tune Small, Align Large" question directly: if you find the best
(lr, beta) on a cheap small model and just apply it, unmodified, to a bigger
target model, how much quality do you give up vs. tuning on the target
directly -- and how does that gap grow as the size difference widens?

This is answerable with ZERO new training. Every size's grid search already
used a SHARED (lr, beta) subgrid (lr in {1e-6, 2e-6, 4e-6}, beta in
{0.01, 0.02, 0.04} -- see COMBINED_LR_VALUES/COMBINED_BETA_VALUES in
common.py, and note 4B's grid has 7 extra points beyond this). So a config
that is "best" on one size's grid is, by construction, also an already-
evaluated point on every other size's grid: transfer regret is a lookup, not
a simulation.

Methodology:
  - Every quantity below (source's own best, target's own best/worst/random
    baseline) is computed on the SHARED 3x3 subgrid only, even for 4B, so
    that "native tuning" means the same thing (a 9-config search) at every
    size and the comparison stays apples-to-apples. 4B's 7 extra grid points
    are ignored here (they're picked up elsewhere, e.g. analyze_results.py).
  - "regret" = target's true best Z-All minus the Z-All of the transferred
    (source-optimal) config, read off the TARGET's own final-fidelity grid.
  - "normalized regret" = regret / (target best Z-All - target worst Z-All)
    on the shared subgrid: 0 = as good as tuning natively on target, 1 = as
    bad as the worst of the 9 shared configs on target.
  - "random regret" (context/baseline) = mean normalized regret of picking
    one of the 9 shared configs uniformly at random on the target -- closed
    form (mean rank), no simulation needed.

Cost view: cost("native" search at target) = 9 x mean_cost(target run).
cost("transfer" strategy) = 9 x mean_cost(source run) [to find source's
best] + 1 x mean_cost(target run) [one confirmatory run at target]. Percent
compute saved by transferring vs. tuning natively at target, paired with the
regret incurred, is the actual "cheapest strategy at fixed budget" plot.

Usage:
    python analyze_transfer_across_scale.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import load_ground_truth_final
from common import COMBINED_BETA_VALUES, COMBINED_LR_VALUES, HERE, SIZE_KEYS, get_paths

# Approximate parameter counts (billions), matching the sizes' own naming.
SIZE_PARAMS_B = {"0.6b": 0.6, "1.7b": 1.7, "4b": 4.0, "8b": 8.0, "14b": 14.0}

TRANSFER_OBJECTIVES = ["Z-All", "Z-Dynamic", "Z-Static"]


def out_dir_for(objective: str) -> Path:
    slug = objective.lower().replace("-", "_")
    d = HERE / "figure" / f"analyze_transfer_across_scale_{slug}"
    d.mkdir(parents=True, exist_ok=True)
    return d


OUT_DIR = out_dir_for("Z-All")  # default; main() overrides per --objective


def shared_grid_final(size_key: str) -> pd.DataFrame:
    """Final-fidelity grid for one size, restricted to the shared 3x3
    (lr, beta) subgrid used by every size (drops 4B's extra sweep points)."""
    paths = get_paths(size_key)
    final, _benchmarks = load_ground_truth_final(paths.grid_csv, size_key)
    mask = final["lr"].isin(COMBINED_LR_VALUES) & final["beta"].isin(COMBINED_BETA_VALUES)
    final = final[mask].copy()
    n_expected = len(COMBINED_LR_VALUES) * len(COMBINED_BETA_VALUES)
    assert len(final) == n_expected, (
        f"{size_key}: expected {n_expected} shared-subgrid configs at final fidelity, got {len(final)}"
    )
    # Re-rank within the shared subgrid only (the grid-wide zall_rank from
    # load_ground_truth_final includes 4B's extra points), one rank column
    # per composite objective so callers can pick which signal to transfer on.
    for objective in TRANSFER_OBJECTIVES:
        final[f"shared_rank_{objective}"] = final[objective].rank(ascending=False, method="min").astype(int)
    return final.reset_index(drop=True)


def mean_full_run_cost_sec(size_key: str) -> float:
    """Mean wall-clock cost of one full DPO training run at this size, over
    the 9 shared-subgrid configs, read from the grid's own elapsed_time_sec
    at final fidelity (cumulative, so this IS the full-run cost)."""
    paths = get_paths(size_key)
    df = pd.read_csv(paths.grid_csv)
    mask = df["lr"].isin(COMBINED_LR_VALUES) & df["beta"].isin(COMBINED_BETA_VALUES)
    df = df[mask]
    final = df[df["dpo_step"] == df["dpo_step"].max()]
    return float(final["elapsed_time_sec"].mean())


def build_transfer_table(objective: str = "Z-All") -> pd.DataFrame:
    grids = {s: shared_grid_final(s) for s in SIZE_KEYS}
    costs = {s: mean_full_run_cost_sec(s) for s in SIZE_KEYS}
    n_shared = len(COMBINED_LR_VALUES) * len(COMBINED_BETA_VALUES)
    rank_col = f"shared_rank_{objective}"

    rows = []
    for source in SIZE_KEYS:
        src = grids[source]
        src_best = src.loc[src[objective].idxmax()]
        for target in SIZE_KEYS:
            tgt = grids[target]
            tgt_best_z = tgt[objective].max()
            tgt_worst_z = tgt[objective].min()
            span = tgt_best_z - tgt_worst_z
            span = span if span > 1e-12 else np.nan

            hit = tgt[(tgt["lr"] == src_best["lr"]) & (tgt["beta"] == src_best["beta"])]
            assert len(hit) == 1, f"transferred config not found on {target}'s shared grid"
            transferred_z = float(hit[objective].iloc[0])
            transferred_rank = int(hit[rank_col].iloc[0])
            regret = tgt_best_z - transferred_z
            norm_regret = regret / span if span == span else 0.0  # NaN-safe: span!=NaN check

            random_norm_regret = (tgt_best_z - tgt[objective].mean()) / span if span == span else 0.0

            native_cost = n_shared * costs[target]
            transfer_cost = n_shared * costs[source] + 1 * costs[target]
            pct_compute_saved = 100.0 * (1.0 - transfer_cost / native_cost)

            rows.append({
                "source": source, "target": target,
                "source_params_b": SIZE_PARAMS_B[source], "target_params_b": SIZE_PARAMS_B[target],
                "log2_size_ratio": np.log2(SIZE_PARAMS_B[target] / SIZE_PARAMS_B[source]),
                "transferred_lr": src_best["lr"], "transferred_beta": src_best["beta"],
                "transferred_rank_of_9": transferred_rank,
                "norm_regret": norm_regret,
                "random_norm_regret": random_norm_regret,
                "native_cost_sec": native_cost, "transfer_cost_sec": transfer_cost,
                "pct_compute_saved": pct_compute_saved,
            })
    return pd.DataFrame(rows)


def plot_heatmap(table: pd.DataFrame, objective: str) -> None:
    mat = np.full((len(SIZE_KEYS), len(SIZE_KEYS)), np.nan)
    for i, source in enumerate(SIZE_KEYS):
        for j, target in enumerate(SIZE_KEYS):
            row = table[(table["source"] == source) & (table["target"] == target)]
            mat[i, j] = row["norm_regret"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SIZE_KEYS)))
    ax.set_xticklabels([f"Qwen3-{s}" for s in SIZE_KEYS], rotation=30, ha="right")
    ax.set_yticks(range(len(SIZE_KEYS)))
    ax.set_yticklabels([f"Qwen3-{s}" for s in SIZE_KEYS])
    ax.set_xlabel("target (applied to)")
    ax.set_ylabel("source (tuned on)")
    ax.set_title(f"Transfer regret: apply source's optimal (lr, beta) to target\n"
                 f"(normalized {objective} regret, 0 = as good as tuning natively)")
    for i in range(len(SIZE_KEYS)):
        for j in range(len(SIZE_KEYS)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                     color="white" if mat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="normalized regret")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "transfer_regret_heatmap.png", dpi=150)
    plt.close(fig)


def plot_regret_vs_gap(table: pd.DataFrame, objective: str) -> None:
    off_diag = table[table["source"] != table["target"]].copy()
    small_to_large = off_diag[off_diag["log2_size_ratio"] > 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(small_to_large["log2_size_ratio"], small_to_large["norm_regret"],
               c="#4C72B0", s=50, alpha=0.8, label="transfer regret (small -> large)", zorder=3)
    # random-baseline reference: mean over targets (a constant-ish band, since it
    # only depends on the target's own grid shape, not on the source).
    rand_by_target = off_diag.groupby("target")["random_norm_regret"].mean()
    ax.axhspan(rand_by_target.min(), rand_by_target.max(), color="grey", alpha=0.2,
               label="random-config baseline (range across targets)")

    for _, r in small_to_large.iterrows():
        ax.annotate(f"{r['source']}->{r['target']}", (r["log2_size_ratio"], r["norm_regret"]),
                    fontsize=6, alpha=0.6, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("log2(target params / source params)  [how far you jump in scale]")
    ax.set_ylabel(f"normalized {objective} regret at target")
    ax.set_title(f"Does transfer regret grow with the size gap? (objective: {objective})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "transfer_regret_vs_gap.png", dpi=150)
    plt.close(fig)


def plot_cost_benefit(table: pd.DataFrame, objective: str) -> None:
    small_to_large = table[table["log2_size_ratio"] > 0].copy()

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sizes_norm = 40 + 20 * small_to_large["log2_size_ratio"]
    sc = ax.scatter(small_to_large["pct_compute_saved"], small_to_large["norm_regret"],
                     c=small_to_large["log2_size_ratio"], cmap="viridis", s=sizes_norm, alpha=0.85)
    for _, r in small_to_large.iterrows():
        ax.annotate(f"{r['source']}->{r['target']}", (r["pct_compute_saved"], r["norm_regret"]),
                    fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="log2(size ratio)")
    ax.set_xlabel("% compute saved vs. tuning natively on target\n(9 source-runs + 1 confirmatory target-run, vs. 9 target-runs)")
    ax.set_ylabel(f"normalized {objective} regret at target")
    ax.set_title(f"Cheapest tuning strategy at fixed budget:\ncompute saved vs. quality given up (objective: {objective})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "transfer_cost_benefit.png", dpi=150)
    plt.close(fig)


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", default="Z-All", choices=TRANSFER_OBJECTIVES)
    args = parser.parse_args()
    objective = args.objective
    OUT_DIR = out_dir_for(objective)

    table = build_transfer_table(objective)
    table.to_csv(OUT_DIR / "transfer_regret_table.csv", index=False)
    plot_heatmap(table, objective)
    plot_regret_vs_gap(table, objective)
    plot_cost_benefit(table, objective)

    small_to_large = table[table["log2_size_ratio"] > 0]
    print(f"[transfer:{objective}] wrote {len(table)}-row table to {OUT_DIR / 'transfer_regret_table.csv'}")
    print(f"[transfer:{objective}] small->large mean normalized regret: {small_to_large['norm_regret'].mean():.3f} "
          f"(random baseline: {small_to_large['random_norm_regret'].mean():.3f})")
    print(f"[transfer:{objective}] small->large mean compute saved: {small_to_large['pct_compute_saved'].mean():.1f}%")


if __name__ == "__main__":
    main()
