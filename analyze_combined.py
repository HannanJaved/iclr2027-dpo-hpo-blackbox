#!/usr/bin/env python3
"""
analyze_combined.py

Analyzes the joint cross-size HPO simulation study (run_combined_simulations.py):
does an optimizer that can pick model size AND DPO hyperparameters together
correctly trade off a bigger model's much higher training cost against its
(usually, but not always) higher achievable quality?

Produces:
  - combined_regret_curves.png    : best-value-so-far vs. simulated wallclock
                                     time, one panel per objective (ELO, Z-All),
                                     one line per optimizer.
  - combined_size_distribution.png: for each objective, which SIZE each
                                     optimizer's runs ended up recommending
                                     (stacked bar, one bar per optimizer).
  - combined_recovers_best.png    : mean true rank (out of 45 configs) and
                                     hit-rate on the global optimum of the
                                     recommended config, by optimizer x objective.

Usage:
    python analyze_combined.py [--budget-tag generous|tight]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import plot_regret_curves
from build_blackbox import add_composite_zscores, fill_gaps, usable_benchmarks
from common import (
    COMBINED_SEARCH_OBJECTIVES,
    DEFAULT_BUDGET_TAG,
    OPTIMIZER_COLORS,
    OPTIMIZERS,
    SIZE_KEYS,
    get_combined_paths,
)

GROUP_COLS = ["size", "lr", "beta"]
SIZE_COLORS = {
    "0.6b": "#4C72B0", "1.7b": "#55A868", "4b": "#C44E52", "8b": "#8172B2", "14b": "#CCB974",
}


def load_ground_truth_final(grid_csv) -> pd.DataFrame:
    df = pd.read_csv(grid_csv)
    benchmarks = usable_benchmarks(df, "combined", group_cols=GROUP_COLS)
    df = fill_gaps(df, benchmarks, "combined", group_cols=GROUP_COLS, peer_cols=["size"])
    df = add_composite_zscores(df, benchmarks)  # pooled across all 45 -> cross-size normalized
    final = df[df["dpo_step"] == df["dpo_step"].max()].copy()
    final["zall_rank"] = final["Z-All"].rank(ascending=False, method="min").astype(int)
    return final.reset_index(drop=True)


def recommended_configs(raw: pd.DataFrame) -> pd.DataFrame:
    idx = raw.groupby(["search_objective", "optimizer", "seed"]).apply(lambda g: g[g.name[0]].idxmax())
    return raw.loc[idx.values, ["search_objective", "optimizer", "seed", "size", "lr", "beta"]].reset_index(drop=True)


def plot_size_distribution(recs: pd.DataFrame, budget_tag: str, figures_dir) -> None:
    objectives = [o for o in COMBINED_SEARCH_OBJECTIVES if o in recs["search_objective"].unique()]
    fig, axes = plt.subplots(1, len(objectives), figsize=(6.5 * len(objectives), 5), squeeze=False)
    axes = axes[0]

    for ax, objective in zip(axes, objectives):
        sub = recs[recs["search_objective"] == objective]
        counts = pd.crosstab(sub["optimizer"], sub["size"]).reindex(index=OPTIMIZERS, columns=SIZE_KEYS, fill_value=0)
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
        bottom = np.zeros(len(OPTIMIZERS))
        for size_key in SIZE_KEYS:
            vals = pct[size_key].to_numpy()
            ax.bar(OPTIMIZERS, vals, bottom=bottom, label=f"Qwen3-{size_key}", color=SIZE_COLORS[size_key])
            bottom += vals
        ax.set_ylabel("% of 25 seeds recommending this size", fontsize=9)
        ax.set_title(f"Search objective: {objective}", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, 100)

    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    title = "Which model size does the joint search recommend?"
    if budget_tag != DEFAULT_BUDGET_TAG:
        title += f" (budget regime: {budget_tag})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = figures_dir / "combined_size_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_recovers_best(recs: pd.DataFrame, final: pd.DataFrame, budget_tag: str, figures_dir) -> None:
    merged = recs.merge(final[GROUP_COLS + ["zall_rank"]], on=GROUP_COLS, how="left")
    objectives = [o for o in COMBINED_SEARCH_OBJECTIVES if o in recs["search_objective"].unique()]
    n_configs = final.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.8 / len(objectives)
    x = np.arange(len(OPTIMIZERS))
    colors = {"ELO": "#4C72B0", "Z-All": "#C44E52"}

    for k, objective in enumerate(objectives):
        sub = merged[merged["search_objective"] == objective]
        mean_rank = sub.groupby("optimizer")["zall_rank"].mean().reindex(OPTIMIZERS)
        hit_rate = sub.groupby("optimizer")["zall_rank"].apply(lambda s: (s == 1).mean() * 100).reindex(OPTIMIZERS)
        off = (k - len(objectives) / 2 + 0.5) * width
        ax1.bar(x + off, mean_rank.values, width=width, label=objective, color=colors.get(objective))
        ax2.bar(x + off, hit_rate.values, width=width, label=objective, color=colors.get(objective))

    ax1.axhline(1.0, color="green", linestyle=":", linewidth=1.2, label="best possible (rank 1)")
    ax1.axhline((n_configs + 1) / 2, color="grey", linestyle=":", linewidth=1.0,
                label=f"random guess (rank {(n_configs + 1) / 2:.1f} of {n_configs})")
    ax1.set_xticks(x)
    ax1.set_xticklabels(OPTIMIZERS, rotation=30)
    ax1.set_ylabel(f"Mean true rank of recommended config\n(1 = best of {n_configs}, lower is better)", fontsize=9)
    ax1.set_title("How good is the config found?", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(OPTIMIZERS, rotation=30)
    ax2.set_ylabel("% of runs that found the TRUE best (size,lr,beta)", fontsize=9)
    ax2.set_title("Hit rate on the global optimum", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    title = f"Combined cross-size search: config quality by optimizer x objective (45-config grid, 25 seeds/cell)"
    if budget_tag != DEFAULT_BUDGET_TAG:
        title += f"\nbudget regime: {budget_tag}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = figures_dir / "combined_recovers_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-tag", default=DEFAULT_BUDGET_TAG, choices=["generous", "tight"])
    args = parser.parse_args()

    paths = get_combined_paths(args.budget_tag)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(paths.simulation_raw_csv)
    final = load_ground_truth_final(paths.grid_csv)
    recs = recommended_configs(raw)

    print(f"Ground-truth ranking (budget regime: {args.budget_tag}), top 10 of 45 by pooled Z-All:")
    print(final.sort_values("zall_rank").head(10)[["size", "lr", "beta", "zall_rank", "Z-All", "ELO"]].to_string(index=False))
    print("\nBottom 5 of 45:")
    print(final.sort_values("zall_rank").tail(5)[["size", "lr", "beta", "zall_rank", "Z-All", "ELO"]].to_string(index=False))
    print("\nBest-ranked config per size:")
    print(final.loc[final.groupby("size")["zall_rank"].idxmin()].sort_values("zall_rank")
          [["size", "lr", "beta", "zall_rank", "Z-All", "ELO"]].to_string(index=False))

    label = "combined" if args.budget_tag == DEFAULT_BUDGET_TAG else f"combined ({args.budget_tag} budget)"
    plot_regret_curves(raw, headline_objectives=COMBINED_SEARCH_OBJECTIVES, size_key=label,
                        figures_dir=paths.figures_dir, x_axis="time")
    plot_size_distribution(recs, args.budget_tag, paths.figures_dir)
    plot_recovers_best(recs, final, args.budget_tag, paths.figures_dir)


if __name__ == "__main__":
    main()
