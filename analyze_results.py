#!/usr/bin/env python3
"""
analyze_results.py

Turns one model size's 500 simulated HPO runs (run_simulations.py) into the
three comparisons the study was set up for:

  1. Optimizer comparison: does multi-fidelity (ASHA / BOHB) reach good
     values faster / cheaper than single-fidelity (RandomSearch / BOTorch)?
     -> regret_curves.png (best-value-so-far vs. simulated wallclock time)

  2. Cross-metric transfer: if you search using objective X (e.g. ELO), how
     good is the config you end up with according to every OTHER metric?
     -> cross_metric_transfer.png (X = search objective, Y = evaluation
        metric, both averaged over optimizers/seeds and expressed as a
        z-score against the true grid at final fidelity)

  3. Which objective finds the best configuration? For each search
     objective, how close (in true Z-All rank, 1 = best config) does the
     recommended config get to the actual optimum?
     -> objective_recovers_best_config.png

For (2) and (3), "the config a run recommends" is defined as the (lr, beta)
that achieved the run's best *observed* value of its search objective (an
online argmax, matching what a practitioner would actually pick). Its quality
on other metrics is then read off the GROUND-TRUTH grid at final fidelity
(i.e. "if this config were trained to completion, how good would it truly
be"), not from whatever partial/early-stopped report happened to be
observed -- this avoids crediting/blaming a search objective for an early
stopping snapshot rather than the configuration itself.

Usage:
    python analyze_results.py --size 1.7b
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_blackbox import add_composite_zscores, fill_gaps, usable_benchmarks
from common import OPTIMIZER_COLORS, OPTIMIZERS, SEARCH_OBJECTIVES, get_paths

ALL_TRANSFER_METRICS = ["ELO", "Arena-Hard", "MT-Bench", "AlpacaEval", "IFEval", "Z-Dynamic", "Z-Static", "Z-All"]


def load_ground_truth_final(grid_csv, size_key: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(grid_csv)
    benchmarks = usable_benchmarks(df, size_key)
    df = add_composite_zscores(fill_gaps(df, benchmarks, size_key), benchmarks)
    final = df[df["dpo_step"] == df["dpo_step"].max()].copy()

    metric_cols = benchmarks + ["Z-Static", "Z-Dynamic", "Z-All"]
    for col in metric_cols:
        std = final[col].std(ddof=0)
        final[f"{col}__z"] = (final[col] - final[col].mean()) / std if std > 1e-12 else 0.0
    final["zall_rank"] = final["Z-All"].rank(ascending=False, method="min").astype(int)
    return final.reset_index(drop=True), benchmarks


def recommended_configs(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (search_objective, optimizer, seed) run, the (lr, beta) with the
    best *observed* value of that run's search objective."""
    idx = raw.groupby(["search_objective", "optimizer", "seed"]).apply(
        lambda g: g[g.name[0]].idxmax()
    )
    return raw.loc[idx.values, ["search_objective", "optimizer", "seed", "lr", "beta"]].reset_index(drop=True)


def plot_regret_curves(raw: pd.DataFrame, headline_objectives: list[str], size_key: str, figures_dir,
                        x_axis: str = "time") -> None:
    """Mean best-value-so-far vs. search budget, one panel per objective.

    :param x_axis: "time" plots against simulated wallclock time (hours) --
        the budget-aware view, where multi-fidelity methods win partly
        *because* a pruned trial's unpaid-for high-fidelity checkpoints save
        real GPU time. "evals" plots against the cumulative number of
        blackbox evaluations (one per (config, checkpoint) report a scheduler
        received) instead -- a budget-agnostic view where every evaluation
        counts as one unit regardless of how expensive that checkpoint
        actually was to train. The two can disagree: a single-fidelity trial
        that always runs to completion contributes all 11 checkpoint reports
        every time, win or lose, while a pruned multi-fidelity trial
        contributes only the few it was allowed before stopping.
    """
    assert x_axis in ("time", "evals")
    n_points = 60
    n = len(headline_objectives)
    ncols = min(3, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[n:]:
        ax.axis("off")

    for ax, objective in zip(axes, headline_objectives):
        sub = raw[raw["search_objective"] == objective]
        if x_axis == "time":
            x_grid = np.linspace(0, sub["st_tuner_time"].max(), n_points)
        else:
            x_grid = np.arange(1, sub.groupby(["optimizer", "seed"]).size().max() + 1)

        for optimizer in OPTIMIZERS:
            opt_sub = sub[sub["optimizer"] == optimizer]
            seed_curves = []
            for seed, g in opt_sub.groupby("seed"):
                g = g.sort_values("st_tuner_time")
                best_so_far = g[objective].cummax().to_numpy()
                x_obs = g["st_tuner_time"].to_numpy() if x_axis == "time" else np.arange(1, len(best_so_far) + 1)
                curve = np.interp(x_grid, x_obs, best_so_far, left=np.nan, right=best_so_far[-1])
                seed_curves.append(curve)
            arr = np.array(seed_curves)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
            color = OPTIMIZER_COLORS[optimizer]
            x_plot = x_grid / 3600 if x_axis == "time" else x_grid
            ax.plot(x_plot, mean, color=color, linewidth=2, label=optimizer)
            ax.fill_between(x_plot, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.set_xlabel("Simulated wallclock time (hours, 2 workers)" if x_axis == "time"
                      else "Cumulative blackbox evaluations\n(config x checkpoint queries)", fontsize=9)
        ax.set_ylabel(f"Best {objective} found so far", fontsize=9)
        ax.set_title(f"Search objective: {objective}", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    budget_desc = "simulated budget" if x_axis == "time" else "number of evaluations"
    fig.suptitle(
        f"Qwen3-{size_key} DPO-AO — optimizer comparison — mean best-value-so-far vs. {budget_desc} "
        "(25 seeds, shaded = ±1 SEM)",
        fontsize=11,
    )
    fig.tight_layout()
    filename = "regret_curves.png" if x_axis == "time" else "regret_curves_by_evals.png"
    out = figures_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_cross_metric_transfer(recs: pd.DataFrame, final: pd.DataFrame, search_objectives: list[str],
                                transfer_metrics: list[str], size_key: str, figures_dir) -> None:
    merged = recs.merge(final[["lr", "beta"] + [f"{m}__z" for m in transfer_metrics]], on=["lr", "beta"], how="left")
    mat = np.zeros((len(search_objectives), len(transfer_metrics)))
    for i, obj in enumerate(search_objectives):
        sub = merged[merged["search_objective"] == obj]
        for j, met in enumerate(transfer_metrics):
            mat[i, j] = sub[f"{met}__z"].mean()

    fig, ax = plt.subplots(figsize=(1.3 * len(transfer_metrics) + 2, 1.1 * len(search_objectives) + 1.5))
    vabs = np.max(np.abs(mat))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-vabs, vmax=vabs, aspect="auto")
    ax.set_xticks(range(len(transfer_metrics)))
    ax.set_xticklabels(transfer_metrics, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(search_objectives)))
    ax.set_yticklabels(search_objectives, fontsize=9)
    ax.set_xlabel("Evaluated on (final-fidelity z-score vs. the config grid)", fontsize=9)
    ax.set_ylabel("Searched for", fontsize=9)
    for i in range(len(search_objectives)):
        for j in range(len(transfer_metrics)):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vabs else "black", fontweight="bold")
    fig.colorbar(im, ax=ax, label="mean z-score of recommended config (over optimizers x seeds)")
    ax.set_title(
        f"Qwen3-{size_key} DPO-AO — cross-metric transfer: search for row, evaluate on column\n"
        "(4 optimizers x 25 seeds per row)",
        fontsize=10,
    )
    fig.tight_layout()
    out = figures_dir / "cross_metric_transfer.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_objective_recovers_best_config(recs: pd.DataFrame, final: pd.DataFrame, search_objectives: list[str],
                                         size_key: str, figures_dir) -> None:
    merged = recs.merge(final[["lr", "beta", "zall_rank"]], on=["lr", "beta"], how="left")
    n_configs = final.shape[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    mean_rank = merged.groupby("search_objective")["zall_rank"].mean().reindex(search_objectives)
    ax1.bar(mean_rank.index, mean_rank.values, color="#4C72B0")
    ax1.axhline(1.0, color="green", linestyle=":", linewidth=1.2, label="best possible (rank 1)")
    ax1.axhline((n_configs + 1) / 2, color="grey", linestyle=":", linewidth=1.0,
                label=f"random guess (rank {(n_configs + 1) / 2:.1f} of {n_configs})")
    ax1.set_ylabel(f"Mean true Z-All rank of recommended config\n(1 = best of {n_configs} configs, lower is better)", fontsize=9)
    ax1.set_title("How good is the config each objective finds?", fontsize=10)
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    hit_rate = merged.groupby("search_objective")["zall_rank"].apply(lambda s: (s == 1).mean() * 100).reindex(search_objectives)
    ax2.bar(hit_rate.index, hit_rate.values, color="#DD8452")
    ax2.set_ylabel("% of runs that found the TRUE best config", fontsize=9)
    ax2.set_title("Hit rate on the global optimum", fontsize=10)
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"Qwen3-{size_key} DPO-AO — which search objective recovers the best DPO config? "
                 f"(100 runs/objective, {n_configs}-config grid)", fontsize=11)
    fig.tight_layout()
    out = figures_dir / "objective_recovers_best_config.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nMean true Z-All rank of recommended config (1=best), by search objective:")
    print(mean_rank.round(2).to_string())
    print("\nHit rate on the true global optimum (%), by search objective:")
    print(hit_rate.round(1).to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--budget-tag", default="generous", choices=["generous", "tight"])
    args = parser.parse_args()

    paths = get_paths(args.size, args.budget_tag)
    # Only used for figure titles (not paths) so "tight" budget plots are
    # unambiguous even if the PNG is viewed out of context.
    label = args.size if args.budget_tag == "generous" else f"{args.size} ({args.budget_tag} budget)"

    paths.figures_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(paths.simulation_raw_csv)
    final, benchmarks = load_ground_truth_final(paths.grid_csv, args.size)
    recs = recommended_configs(raw)

    # Only objectives this size's blackbox actually has (run_simulations.py
    # already skipped any that were dropped for missing eval data).
    search_objectives = [o for o in SEARCH_OBJECTIVES if o in raw["search_objective"].unique()]
    transfer_metrics = [m for m in ALL_TRANSFER_METRICS if m in benchmarks or m in ("Z-Static", "Z-Dynamic", "Z-All")]

    print(f"Ground-truth ranking for Qwen3-{args.size} (final fidelity, by Z-All):")
    show_cols = ["lr", "beta", "zall_rank", "Z-All"] + [m for m in ("ELO", "Arena-Hard") if m in benchmarks]
    print(final.sort_values("zall_rank")[show_cols].to_string(index=False))

    # Z-All first (the composite reference), then every other proxy that was
    # actually searched for this size.
    ordered_objectives = ["Z-All"] + [o for o in search_objectives if o != "Z-All"]
    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths.figures_dir, x_axis="time")
    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths.figures_dir, x_axis="evals")
    plot_cross_metric_transfer(recs, final, search_objectives, transfer_metrics, size_key=label, figures_dir=paths.figures_dir)
    plot_objective_recovers_best_config(recs, final, search_objectives, size_key=label, figures_dir=paths.figures_dir)


if __name__ == "__main__":
    main()
