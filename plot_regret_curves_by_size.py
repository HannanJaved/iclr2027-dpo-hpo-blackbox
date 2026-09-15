#!/usr/bin/env python3
"""
plot_regret_curves_by_size.py

Cross-size counterpart to analyze_results_surrogate.py's regret curves: instead
of one panel per search objective (all optimizers overlaid, single model size --
see analyze_results.py's plot_regret_curves), this produces ONE figure with one
panel PER MODEL SIZE, for a single surrogate and a single y-axis metric.

All runs are the Z-All-optimizing search (the only composite objective actually
searched over, see common.SEARCH_OBJECTIVES). Two y-axis metrics are supported:
  - Z-All:     the native optimization curve (best Z-All found so far).
  - Z-Dynamic: a transfer view -- the SAME Z-All-optimizing runs, but tracking
    best-Z-Dynamic-found-so-far instead. Shows how well optimizing the full
    11-benchmark composite transfers to the 4-benchmark chat/preference subset.

Produces one PNG per (surrogate, y-metric) pair, e.g.:
    figures_by_size/gp/regret_by_size_Z-All.png
    figures_by_size/gp/regret_by_size_Z-Dynamic.png
(plus an "_by_evals" variant of each, budget-agnostic x-axis).

Usage:
    python plot_regret_curves_by_size.py --surrogate gp
    python plot_regret_curves_by_size.py --surrogate all
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_blackbox_surrogate import SURROGATE_CHOICES, get_surrogate_results_paths  # noqa: E402
from common import DEFAULT_FAMILY, FAMILIES, FAMILY_LABELS, HERE, OPTIMIZER_COLORS, OPTIMIZERS, SIZE_KEYS_BY_FAMILY, get_paths  # noqa: E402

BASE_SEARCH_OBJECTIVE = "Z-All"  # the only composite objective actually searched over
Y_METRICS = ["Z-All", "Z-Dynamic"]
N_POINTS = 60


def _simulation_raw_csv_path(size_key: str, surrogate: str, family: str):
    """knn1 predates the --surrogate refactor: its data was produced by the
    very first version of build_blackbox_surrogate.py / run_simulations_surrogate.py,
    which had no --surrogate flag and wrote to "results_surrogate/" (no
    "-knn1" suffix), unlike knn3/knn5/gp which all use the current
    get_surrogate_results_paths() naming. Route knn1 to that legacy path."""
    if surrogate == "knn1":
        size_dir = get_paths(size_key, family=family).size_dir
        return size_dir / "results_surrogate" / "simulation_raw.csv"
    return get_surrogate_results_paths(size_key, family=family, surrogate=surrogate)["simulation_raw_csv"]


def plot_one(size_keys: list[str], surrogate: str, family: str,
             y_metric: str, x_axis: str, figures_dir) -> None:
    assert x_axis in ("time", "evals")
    family_label = FAMILY_LABELS.get(family, family.capitalize())

    ncols = min(3, len(size_keys))
    nrows = -(-len(size_keys) // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    axes = axes.flatten()
    for ax in axes[len(size_keys):]:
        ax.axis("off")

    any_data = False
    for ax, size_key in zip(axes, size_keys):
        csv_path = _simulation_raw_csv_path(size_key, surrogate, family)
        if not csv_path.exists():
            ax.axis("off")
            ax.text(0.5, 0.5, f"{size_key}: no data", ha="center", va="center", fontsize=16, color="gray")
            continue
        raw = pd.read_csv(csv_path)
        sub = raw[raw["search_objective"] == BASE_SEARCH_OBJECTIVE]
        if sub.empty or y_metric not in sub.columns:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{size_key}: no data", ha="center", va="center", fontsize=16, color="gray")
            continue
        any_data = True

        if x_axis == "time":
            x_grid = np.linspace(0, sub["st_tuner_time"].max(), N_POINTS)
        else:
            x_grid = np.arange(1, sub.groupby(["optimizer", "seed"]).size().max() + 1)

        for optimizer in OPTIMIZERS:
            opt_sub = sub[sub["optimizer"] == optimizer]
            seed_groups = list(opt_sub.groupby("seed"))

            # A mean over a set of seeds that grows partway through a curve
            # can dip even though every individual seed's own cummax curve
            # is non-decreasing -- whenever a newly-joining seed's value is
            # below the running average, the mean drops at that instant. The
            # only way to guarantee a monotonic mean is to fix the seed set
            # up front and never change it. For x_axis="time", first-report
            # times are staggered across seeds, so: freeze the ~90%-earliest
            # seeds (by first report time) as the ones this curve will ever
            # use, permanently dropping the slowest stragglers for this
            # optimizer/size -- rather than waiting for literally every seed
            # (which lets a single straggler delay the whole curve's start,
            # e.g. one seed at 4.8h vs. the rest at 1.9h for 14b/BOHB) or
            # admitting seeds gradually (which reintroduces the same dip).
            # For x_axis="evals" every seed already has a value at the first
            # grid point (cumulative eval count, not wall-clock time), so
            # there is no staggering and no seeds need to be dropped.
            if x_axis == "time":
                first_times = {seed: g["st_tuner_time"].min() for seed, g in seed_groups}
                n_thresh = int(np.ceil(0.9 * len(seed_groups)))
                included_seeds = set(sorted(first_times, key=first_times.get)[:n_thresh])
            else:
                included_seeds = {seed for seed, _ in seed_groups}

            seed_curves = []
            for seed, g in seed_groups:
                if seed not in included_seeds:
                    continue
                g = g.sort_values("st_tuner_time")
                best_so_far = g[y_metric].cummax().to_numpy()
                x_obs = g["st_tuner_time"].to_numpy() if x_axis == "time" else np.arange(1, len(best_so_far) + 1)
                curve = np.interp(x_grid, x_obs, best_so_far, left=np.nan, right=best_so_far[-1])
                seed_curves.append(curve)
            arr = np.array(seed_curves)
            n_valid = np.sum(~np.isnan(arr), axis=0)
            # Once every seed in the frozen subset has reported, n_valid
            # stays at arr.shape[0] for the rest of the curve (no more seeds
            # ever join), so this mask only trims the leading edge.
            partial_coverage = n_valid < arr.shape[0]
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0, ddof=0) / np.sqrt(n_valid.clip(min=1))
            mean[partial_coverage] = np.nan
            sem[partial_coverage] = np.nan
            color = OPTIMIZER_COLORS[optimizer]
            x_plot = x_grid / 3600 if x_axis == "time" else x_grid
            ax.plot(x_plot, mean, color=color, linewidth=2.2, label=optimizer)
            ax.fill_between(x_plot, mean - sem, mean + sem, color=color, alpha=0.15)

        ax.set_xlabel("Simulated wallclock time (hours, 2 workers)" if x_axis == "time"
                      else "Cumulative blackbox evaluations\n(config × checkpoint queries)", fontsize=15)
        ax.set_ylabel(f"Best {y_metric} found so far", fontsize=15)
        ax.set_title(f"{family_label}-{size_key}", fontsize=17)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=13)

    if not any_data:
        plt.close(fig)
        print(f"[SKIP] surrogate={surrogate} y_metric={y_metric} x_axis={x_axis}: no data for any size")
        return

    budget_desc = "simulated budget" if x_axis == "time" else "number of evaluations"
    transfer_note = "" if y_metric == BASE_SEARCH_OBJECTIVE else \
        f" (transfer: optimize {BASE_SEARCH_OBJECTIVE}, track {y_metric})"
    fig.suptitle(
        f"{family_label} DPO-AO — {surrogate} surrogate across sizes\n"
        f"best {y_metric} vs {budget_desc}{transfer_note} (25 seeds, ±1 SEM)",
        fontsize=18,
    )
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if x_axis == "time" else "_by_evals"
    filename = f"regret_by_size_{y_metric}{suffix}.png"
    out = figures_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    parser.add_argument("--surrogate", default="all", choices=SURROGATE_CHOICES + ["all"])
    args = parser.parse_args()

    surrogates = SURROGATE_CHOICES if args.surrogate == "all" else [args.surrogate]
    size_keys = SIZE_KEYS_BY_FAMILY[args.family]
    figures_root = HERE / "figures_by_size"

    for surrogate in surrogates:
        figures_dir = figures_root / surrogate
        for y_metric in Y_METRICS:
            for x_axis in ("time", "evals"):
                plot_one(size_keys, surrogate, args.family, y_metric, x_axis, figures_dir)


if __name__ == "__main__":
    main()
