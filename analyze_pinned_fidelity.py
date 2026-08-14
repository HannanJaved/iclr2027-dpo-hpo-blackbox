#!/usr/bin/env python3
"""
analyze_pinned_fidelity.py

Analyzes run_pinned_fidelity_simulations.py's output: the 4 single-fidelity
methods from the general study (RandomSearch, BOTorch, TPE, CQR, unchanged --
included so this sweep gives the full picture on Z-Dynamic, which wasn't part
of the general study's search objectives) plus three ASHA/BOHB variants, all
starting no earlier than step 400: -step400 (aggressive ~1% keep-rate, one
decision, then runs to completion unsupervised), -min400 (gentle ~50%
keep-rate, keeps checking again at 800/1600), and -step400-gentle (gentle
~50% keep-rate like -min400, but only ever checks once, like -step400) --
the third variant isolates whether -min400's worse showing came from
re-checking after 400, or from being less selective at 400 in the first
place. Across Z-Dynamic and Z-All, all 5 sizes. Two views:

  - regret_curves_pinned400.png : best-value-so-far vs. simulated wallclock
    time (mean over 25 seeds), one row per objective, one panel per size --
    same style as run_simulations.py's regret_curves.png, for direct visual
    comparison against the *default* multi-rung ASHA/BOHB schedule.
  - tight_budget_summary.csv/.png : mean % of each size's eventual best value
    reached by the "tight" budget cutoff (1x one full-fidelity run's cost,
    same definition as common.BUDGET_REGIMES["tight"]) -- the single-number
    summary that answers "does a real async scheduler, told only to check in
    at step 400, actually find better configs sooner than blind full-budget
    search?"

Usage:
    python analyze_pinned_fidelity.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BUDGET_REGIMES, HERE, OPTIMIZER_COLORS, SIZE_KEYS, slug

OBJECTIVES = ["Z-Dynamic", "Z-All"]
OPTIMIZERS = [
    "RandomSearch", "BOTorch", "TPE", "CQR",
    "ASHA-step400", "BOHB-step400",
    "ASHA-min400", "BOHB-min400",
    "ASHA-step400-gentle", "BOHB-step400-gentle",
]
# Reuse the study's established per-optimizer colors (common.OPTIMIZER_COLORS)
# so these plots read consistently with every other figure. All ASHA variants
# share the orange hue family, all BOHB variants share red, so they read as
# related at a glance; shade distinguishes which of the two axes changed:
#   step400        : rf=100 (~1% keep), single rung           -> base color
#   min400         : rf=2   (~50% keep), rungs at 400/800/1600 -> darkest shade
#   step400-gentle : rf=2   (~50% keep), single rung           -> lightest shade
# (step400-gentle isolates "does re-checking after 400 hurt" from "is the
# 400 cutoff itself too harsh," which step400 vs. min400 alone conflate.)
COLORS = {
    **OPTIMIZER_COLORS,
    "ASHA-step400": OPTIMIZER_COLORS["ASHA"], "ASHA-min400": "#8C4A1E", "ASHA-step400-gentle": "#F5B77E",
    "BOHB-step400": OPTIMIZER_COLORS["BOHB"], "BOHB-min400": "#7A2E31", "BOHB-step400-gentle": "#E08589",
}

OUT_DIR = HERE / "figure" / "analyze_pinned_fidelity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_size(size_key: str) -> pd.DataFrame | None:
    f = HERE / f"qwen3_{slug(size_key)}_dpo" / "results_pinned400" / "simulation_raw.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


def plot_regret_curves(optimizers: list[str] = OPTIMIZERS, filename: str = "regret_curves_pinned400.png",
                        suptitle: str = "All 6 optimizers, ASHA/BOHB with a single rung pinned at step 400") -> None:
    available = [s for s in SIZE_KEYS if load_size(s) is not None]
    fig, axes = plt.subplots(len(OBJECTIVES), len(available), figsize=(3.6 * len(available), 3.2 * len(OBJECTIVES)),
                              squeeze=False)
    for row, objective in enumerate(OBJECTIVES):
        for col, size_key in enumerate(available):
            ax = axes[row][col]
            raw = load_size(size_key)
            sub = raw[raw["search_objective"] == objective]
            if sub.empty:
                ax.set_visible(False)
                continue
            for optimizer in optimizers:
                g = sub[sub["optimizer"] == optimizer]
                # mean best-so-far across seeds, on a common time grid
                grids = []
                for seed, gs in g.groupby("seed"):
                    gs = gs.sort_values("st_tuner_time")
                    grids.append(gs.set_index("st_tuner_time")[objective].cummax())
                if not grids:
                    continue
                all_times = np.unique(np.concatenate([s.index.values for s in grids]))
                aligned = pd.DataFrame({i: s.reindex(all_times).ffill() for i, s in enumerate(grids)})
                mean = aligned.mean(axis=1)
                sem = aligned.std(axis=1) / np.sqrt(aligned.shape[1])
                hours = all_times / 3600
                ax.plot(hours, mean, color=COLORS[optimizer], label=optimizer, linewidth=1.6)
                ax.fill_between(hours, mean - sem, mean + sem, color=COLORS[optimizer], alpha=0.15)
            ax.set_title(f"Qwen3-{size_key}" if row == 0 else "", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{objective}\nbest so far")
            if row == len(OBJECTIVES) - 1:
                ax.set_xlabel("hours")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"{suptitle} (25 seeds, shaded=±SEM)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=150)
    plt.close(fig)


# Selectivity-vs-schedule comparison, uncluttered: drops the aggressive
# single-rung -step400 pair (the headline result, already shown on its own)
# and BOTorch (noisiest single-fidelity line), leaving RandomSearch as the
# no-early-stopping reference against the three gentle-cutoff variants
# (-min400 checks again at 800/1600, -step400-gentle checks only once) --
# the pair that isolates whether re-checking after 400 matters at all.
GENTLE_COMPARISON_OPTIMIZERS = [
    "RandomSearch", "TPE", "CQR",
    "ASHA-min400", "BOHB-min400",
    "ASHA-step400-gentle", "BOHB-step400-gentle",
]

# Minimal 3-way view: just the no-early-stopping reference against the two
# min400 schedulers (gentle cutoff at 400, keeps checking at 800/1600).
MIN400_ONLY_OPTIMIZERS = ["RandomSearch", "ASHA-min400", "BOHB-min400"]


def plot_tight_budget_bars(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(OBJECTIVES), figsize=(6.5 * len(OBJECTIVES), 4.5), squeeze=False)
    x = np.arange(len(SIZE_KEYS))
    width = 0.8 / len(OPTIMIZERS)
    for col, objective in enumerate(OBJECTIVES):
        ax = axes[0][col]
        sub = summary[summary["objective"] == objective]
        piv = sub.pivot(index="size", columns="optimizer", values="pct_of_best_ever_at_tight_budget").reindex(SIZE_KEYS)
        for i, optimizer in enumerate(OPTIMIZERS):
            if optimizer not in piv.columns:
                continue
            ax.bar(x + (i - (len(OPTIMIZERS) - 1) / 2) * width, piv[optimizer], width,
                   label=optimizer, color=COLORS[optimizer])
        ax.set_xticks(x)
        ax.set_xticklabels([f"Qwen3-{s}" for s in SIZE_KEYS], rotation=20)
        ax.set_ylabel("% of eventual best reached\nat tight-budget cutoff")
        ax.set_title(objective)
        if col == 0:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Tight-budget snapshot: all 6 optimizers, ASHA/BOHB pinned at step 400")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tight_budget_bars.png", dpi=150)
    plt.close(fig)


def tight_budget_summary() -> pd.DataFrame:
    rows = []
    for objective in OBJECTIVES:
        for size_key in SIZE_KEYS:
            raw = load_size(size_key)
            if raw is None:
                continue
            sub = raw[raw["search_objective"] == objective]
            if sub.empty:
                continue
            cutoff = sub["st_tuner_time"].max() / BUDGET_REGIMES["generous"] * BUDGET_REGIMES["tight"]
            best_ever = sub[objective].max()
            for optimizer, g in sub.groupby("optimizer"):
                vals = []
                for seed, gs in g.groupby("seed"):
                    gs = gs.sort_values("st_tuner_time")
                    before = gs[gs["st_tuner_time"] <= cutoff]
                    v = before[objective].cummax().iloc[-1] if not before.empty else np.nan
                    vals.append(v)
                vals = np.array(vals, dtype=float)
                rows.append({
                    "objective": objective, "size": size_key, "optimizer": optimizer,
                    "pct_of_best_ever_at_tight_budget": 100 * np.nanmean(vals) / best_ever if best_ever else np.nan,
                })
    return pd.DataFrame(rows)


def main() -> None:
    plot_regret_curves()
    plot_regret_curves(
        optimizers=GENTLE_COMPARISON_OPTIMIZERS,
        filename="regret_curves_gentle_comparison.png",
        suptitle="Gentle-cutoff comparison: does re-checking after step 400 matter? "
                  "(RandomSearch vs. -min400 vs. -step400-gentle; -step400 and BOTorch omitted)",
    )
    plot_regret_curves(
        optimizers=MIN400_ONLY_OPTIMIZERS,
        filename="regret_curves_min400_only.png",
        suptitle="RandomSearch vs. ASHA/BOHB-min400 (gentle cutoff at 400, keeps checking at 800/1600)",
    )
    summary = tight_budget_summary()
    summary.to_csv(OUT_DIR / "tight_budget_summary.csv", index=False)
    plot_tight_budget_bars(summary)
    for objective in OBJECTIVES:
        sub = summary[summary["objective"] == objective]
        piv = sub.pivot(index="size", columns="optimizer", values="pct_of_best_ever_at_tight_budget").reindex(SIZE_KEYS)
        piv = piv[[o for o in OPTIMIZERS if o in piv.columns]]
        print(f"--- {objective}: % of eventual best reached at tight-budget cutoff ---")
        print(piv.round(1))
        print("pooled mean:", piv.mean(axis=0).round(1).to_dict())
        print()


if __name__ == "__main__":
    main()
