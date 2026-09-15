#!/usr/bin/env python3
"""
analyze_results_surrogate.py

Regret-curves-only counterpart to analyze_results.py, for the continuous
surrogate study (run_simulations_surrogate.py). Reuses plot_regret_curves
directly from analyze_results.py -- it only needs the raw per-report
simulation table (search_objective/optimizer/seed/st_tuner_time/objective
columns), which has the same shape here as in the discrete study.

Deliberately does NOT port cross_metric_transfer or
objective_recovers_best_config: both need a ground-truth final-fidelity
ranking (load_ground_truth_final, keyed off the discrete grid's cfg_id/
config_map), which doesn't apply the same way to the continuous space -- not
built unless asked for.

Usage:
    python analyze_results_surrogate.py --size 4b --surrogate gp
    python analyze_results_surrogate.py --size 8b --surrogate knn5
"""
from __future__ import annotations

import argparse

import pandas as pd

from analyze_results import plot_regret_curves
from build_blackbox_surrogate import get_surrogate_results_paths
from common import DEFAULT_FAMILY, FAMILIES, FAMILY_LABELS, SEARCH_OBJECTIVES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", required=True)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=FAMILIES)
    parser.add_argument("--surrogate", default="gp", choices=["knn1", "knn3", "knn5", "gp"])
    args = parser.parse_args()

    paths = get_surrogate_results_paths(args.size, family=args.family, surrogate=args.surrogate)
    family_label = FAMILY_LABELS.get(args.family, args.family.capitalize())
    label = f"{family_label}-{args.size} (surrogate, {args.surrogate})"

    paths["figures_dir"].mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(paths["simulation_raw_csv"])

    search_objectives = [o for o in SEARCH_OBJECTIVES if o in raw["search_objective"].unique()]
    ordered_objectives = ["Z-All"] + [o for o in search_objectives if o != "Z-All"]

    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths["figures_dir"], x_axis="time")
    plot_regret_curves(raw, headline_objectives=ordered_objectives, size_key=label,
                        figures_dir=paths["figures_dir"], x_axis="evals")


if __name__ == "__main__":
    main()
