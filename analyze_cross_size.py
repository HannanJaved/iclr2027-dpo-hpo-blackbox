#!/usr/bin/env python3
"""
analyze_cross_size.py

Combines the five independent per-size HPO simulation studies
(qwen3_<size>_dpo/, one blackbox and 400-900 simulated runs each) into a
single cross-size view. Configs are never mixed *across* sizes for the HPO
search itself -- each size keeps its own blackbox, grid, and simulations
(see run_simulations.py). This script only lines up the summary statistics
computed per size, side by side, to answer: does "which proxy metric works
best" or "does multi-fidelity help" generalize across model scale, or is it
size-specific?

Produces (under results_cross_size/, or results_cross_size_<tag>/ for a
non-default --budget-tag):
  - cross_size_hit_rate.png   : heatmap, rows=size, cols=search objective,
                                 value=% of runs recovering the TRUE best config
  - cross_size_mean_rank.png  : heatmap, rows=size, cols=search objective,
                                 value=normalized rank of recommended config
                                 (0=best possible, 0.5=random guess, 1=worst).
                                 Normalized (rank-1)/(n_configs-1) so 4B's
                                 16-config grid is comparable to the other
                                 sizes' 9-config grids.
  - cross_size_mf_benefit.png : grouped bar, one group per size, bars=optimizer,
                                 % of the 25 seeds that reached the top decile
                                 of that size's OWN grid (by Z-All) within
                                 budget. Z-All is used because it's the only
                                 objective present at every size (AlpacaEval
                                 was dropped for 14B, see build_blackbox.py).

Usage:
    python analyze_cross_size.py [--budget-tag generous|tight]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_results import load_ground_truth_final, recommended_configs
from common import DEFAULT_BUDGET_TAG, HERE, OPTIMIZER_COLORS, OPTIMIZERS, SEARCH_OBJECTIVES, SIZE_KEYS, get_paths


def cross_size_dir(budget_tag: str) -> Path:
    dirname = "results_cross_size" if budget_tag == DEFAULT_BUDGET_TAG else f"results_cross_size_{budget_tag}"
    return HERE / dirname


def per_size_stats(budget_tag: str) -> pd.DataFrame:
    rows = []
    for size_key in SIZE_KEYS:
        paths = get_paths(size_key, budget_tag)
        raw = pd.read_csv(paths.simulation_raw_csv)
        final, _benchmarks = load_ground_truth_final(paths.grid_csv, size_key)
        recs = recommended_configs(raw)
        n_configs = final.shape[0]
        merged = recs.merge(final[["lr", "beta", "zall_rank"]], on=["lr", "beta"], how="left")
        for objective in SEARCH_OBJECTIVES:
            sub = merged[merged["search_objective"] == objective]
            if sub.empty:
                continue  # objective wasn't searchable for this size (missing eval data)
            mean_rank = sub["zall_rank"].mean()
            hit_rate = (sub["zall_rank"] == 1).mean() * 100
            norm_rank = (mean_rank - 1) / (n_configs - 1) if n_configs > 1 else 0.0
            rows.append({
                "size": size_key, "objective": objective, "n_configs": n_configs,
                "mean_rank": mean_rank, "norm_rank": norm_rank, "hit_rate": hit_rate,
            })
    return pd.DataFrame(rows)


def plot_heatmap(stats: pd.DataFrame, value_col: str, title: str, cbar_label: str,
                  cmap: str, vmin: float, vmax: float, fmt: str, filename: str, out_dir: Path) -> None:
    mat = np.full((len(SIZE_KEYS), len(SEARCH_OBJECTIVES)), np.nan)
    for i, size_key in enumerate(SIZE_KEYS):
        for j, objective in enumerate(SEARCH_OBJECTIVES):
            row = stats[(stats["size"] == size_key) & (stats["objective"] == objective)]
            if not row.empty:
                mat[i, j] = row[value_col].iloc[0]

    fig, ax = plt.subplots(figsize=(1.5 * len(SEARCH_OBJECTIVES) + 2, 1.1 * len(SIZE_KEYS) + 1.5))
    masked = np.ma.masked_invalid(mat)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#dddddd")
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(SEARCH_OBJECTIVES)))
    ax.set_xticklabels(SEARCH_OBJECTIVES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(SIZE_KEYS)))
    ax.set_yticklabels([f"Qwen3-{s}" for s in SIZE_KEYS], fontsize=9)
    for i in range(len(SIZE_KEYS)):
        for j in range(len(SEARCH_OBJECTIVES)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="#777777")
                continue
            frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "white" if frac > 0.6 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9, color=color, fontweight="bold")
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    out = out_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def mf_benefit_stats(budget_tag: str, target_quantile: float = 0.9) -> pd.DataFrame:
    """For the Z-All objective (present at every size), % of seeds per
    optimizer that reach a config at least as good as the target_quantile
    point of THIS size's own grid (worst -> best), within the simulated
    budget. Using a per-size quantile (rather than a fixed raw Z-All value)
    keeps the target comparable across grids of different spread/size."""
    rows = []
    for size_key in SIZE_KEYS:
        paths = get_paths(size_key, budget_tag)
        raw = pd.read_csv(paths.simulation_raw_csv)
        final, _benchmarks = load_ground_truth_final(paths.grid_csv, size_key)
        zall_sorted = final["Z-All"].sort_values()
        worst, best = zall_sorted.iloc[0], zall_sorted.iloc[-1]
        target = worst + target_quantile * (best - worst)

        sub = raw[raw["search_objective"] == "Z-All"]
        for optimizer in OPTIMIZERS:
            opt_sub = sub[sub["optimizer"] == optimizer]
            n_seeds = opt_sub["seed"].nunique()
            n_reached = sum(1 for _seed, g in opt_sub.groupby("seed") if g["Z-All"].max() >= target)
            rows.append({
                "size": size_key, "optimizer": optimizer,
                "pct_reached": 100 * n_reached / n_seeds if n_seeds else np.nan,
            })
    return pd.DataFrame(rows)


def plot_mf_benefit(stats: pd.DataFrame, budget_tag: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    n_opt = len(OPTIMIZERS)
    width = 0.8 / n_opt
    x = np.arange(len(SIZE_KEYS))
    for k, optimizer in enumerate(OPTIMIZERS):
        vals = [
            stats[(stats["size"] == s) & (stats["optimizer"] == optimizer)]["pct_reached"].iloc[0]
            for s in SIZE_KEYS
        ]
        ax.bar(x + (k - n_opt / 2 + 0.5) * width, vals, width=width, label=optimizer, color=OPTIMIZER_COLORS[optimizer])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-{s}" for s in SIZE_KEYS])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of 25 seeds reaching the top decile of the grid\n(by Z-All) within budget", fontsize=9)
    title = "Multi-fidelity benefit across model sizes (search objective: Z-All)"
    if budget_tag != DEFAULT_BUDGET_TAG:
        title += f"\nbudget regime: {budget_tag}"
    ax.set_title(title, fontsize=11)
    # Bars sit near the ceiling in most groups, so there's no reliable empty
    # spot inside the axes for the legend -- put it below the chart instead.
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(OPTIMIZERS), frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "cross_size_mf_benefit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_cross_size_regret_curves(budget_tag: str, out_dir: Path) -> None:
    """Z-All regret curves (best-value-so-far vs. simulated wallclock time),
    one panel per model size, with all 6 optimizers kept distinct (not
    pooled) within each panel -- same style as the per-size regret_curves.png,
    but faceted by size instead of by search objective (Z-All is the only
    objective shown here, since it's the only one present at every size).
    Sizes are NOT mixed for the search itself -- each panel's curves come
    from that size's own independent blackbox and simulations; this just
    lays the panels out together so the optimizer ranking can be compared
    across sizes at a glance.
    """
    n_points = 60
    n = len(SIZE_KEYS)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, size_key in zip(axes, SIZE_KEYS):
        paths = get_paths(size_key, budget_tag)
        raw = pd.read_csv(paths.simulation_raw_csv)
        sub = raw[raw["search_objective"] == "Z-All"]
        max_time = sub["st_tuner_time"].max()
        time_grid = np.linspace(0, max_time, n_points)

        for optimizer in OPTIMIZERS:
            opt_sub = sub[sub["optimizer"] == optimizer]
            seed_curves = []
            for _seed, g in opt_sub.groupby("seed"):
                g = g.sort_values("st_tuner_time")
                best_so_far = g["Z-All"].cummax().to_numpy()
                curve = np.interp(time_grid, g["st_tuner_time"].to_numpy(), best_so_far,
                                   left=np.nan, right=best_so_far[-1])
                seed_curves.append(curve)
            arr = np.array(seed_curves)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
            color = OPTIMIZER_COLORS[optimizer]
            ax.plot(time_grid / 3600, mean, color=color, linewidth=2, label=optimizer)
            ax.fill_between(time_grid / 3600, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.set_xlabel("Simulated wallclock time (hours, 2 workers)", fontsize=9)
        ax.set_ylabel("Best Z-All found so far", fontsize=9)
        ax.set_title(f"Qwen3-{size_key}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=7)

    title = "Z-All regret curves by model size (25 seeds/optimizer, shaded = +/-1 SEM)"
    if budget_tag != DEFAULT_BUDGET_TAG:
        title += f" -- budget regime: {budget_tag}"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = out_dir / "cross_size_regret_curves_zall.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-tag", default=DEFAULT_BUDGET_TAG, choices=["generous", "tight"])
    args = parser.parse_args()
    out_dir = cross_size_dir(args.budget_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = per_size_stats(args.budget_tag)
    stats.to_csv(out_dir / "cross_size_stats.csv", index=False)
    print(stats.to_string(index=False))

    title_suffix = "" if args.budget_tag == DEFAULT_BUDGET_TAG else f"\nbudget regime: {args.budget_tag}"
    plot_heatmap(stats, "hit_rate", f"Hit rate on the TRUE best config, by size x search objective{title_suffix}",
                 "% of runs finding the global optimum", "RdYlGn", 0, 100, ".0f", "cross_size_hit_rate.png", out_dir)
    plot_heatmap(stats, "norm_rank",
                 "Normalized rank of recommended config, by size x search objective\n"
                 f"(0 = best possible, 0.5 = random guess, 1 = worst){title_suffix}",
                 "normalized true Z-All rank (lower = better)", "RdYlGn_r", 0, 1, ".2f",
                 "cross_size_mean_rank.png", out_dir)
    plot_cross_size_regret_curves(args.budget_tag, out_dir)

    mf_stats = mf_benefit_stats(args.budget_tag)
    mf_stats.to_csv(out_dir / "cross_size_mf_benefit.csv", index=False)
    plot_mf_benefit(mf_stats, args.budget_tag, out_dir)


if __name__ == "__main__":
    main()
