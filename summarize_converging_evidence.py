#!/usr/bin/env python3
"""
summarize_converging_evidence.py

RQ3's core claim -- "an early checkpoint of the target model beats transfer
from a smaller model" -- is currently supported by evidence scattered across
several scripts and two genuinely different methodologies. This script pulls
the two triangulating pieces into one small, citable table:

1. Within-size fidelity signal (analyze_transfer_across_fidelity.py's data
   source, extended here with a second statistic): at step 400 (~20% of a
   2031-step DPO run), how well does the FULL ranking of the 9 shared configs
   agree with their final-step ranking (Spearman rho), alongside the
   already-computed argmax-transfer regret. Two different statistics
   (whole-ranking correlation vs. single-point regret) computed from the same
   data, so agreement between them is a real (if modest) sanity check on
   the config-argmax result specifically, not just restating it.

2. Independent joint size+HP search (qwen3_combined_dpo/results/best_found.csv,
   already simulated by run_combined_simulations.py + analyze_combined.py,
   not re-run here): pooled across 6 optimizers x 2 search objectives x 25
   seeds = 300 simulated searches that were free to pick model size AND
   (lr, beta) together, how often do they recommend the two smallest sizes
   versus the two largest? This is a fully independent method (a real
   adaptive search, not an argmax-lookup) reaching a compatible conclusion:
   if smaller sizes were a cheap way to find a transferable config, an
   optimizer hunting for the best config-per-dollar should sometimes land
   there. It essentially never does.

Produces figure/converging_evidence/{fidelity_triangulation.csv, joint_search_size_distribution.csv}.

Usage:
    python summarize_converging_evidence.py
"""
from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr

from analyze_transfer_across_fidelity import (
    EARLY_STEP_HIGHLIGHT,
    TRANSFER_OBJECTIVES,
    build_fidelity_table,
    shared_grid_all_steps,
)
from common import HERE, SIZE_KEYS

OUT_DIR = HERE / "figure" / "converging_evidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fidelity_triangulation() -> pd.DataFrame:
    rho_rows = []
    for size_key in SIZE_KEYS:
        df = shared_grid_all_steps(size_key)
        final_step = df["dpo_step"].max()
        final = df[df["dpo_step"] == final_step].set_index(["lr", "beta"])
        proxy = df[df["dpo_step"] == EARLY_STEP_HIGHLIGHT].set_index(["lr", "beta"])
        for objective in TRANSFER_OBJECTIVES:
            rho, _ = spearmanr(proxy[objective], final.loc[proxy.index, objective])
            rho_rows.append({"size": size_key, "objective": objective, "spearman_rho_step400_vs_final": rho})
    rho_df = pd.DataFrame(rho_rows)

    regret_frames = []
    for objective in TRANSFER_OBJECTIVES:
        t = build_fidelity_table(objective)
        t = t[t["step_s"] == EARLY_STEP_HIGHLIGHT][["size", "transferred_rank_of_9", "norm_regret", "pct_compute_saved"]]
        t["objective"] = objective
        regret_frames.append(t)
    regret_df = pd.concat(regret_frames, ignore_index=True)

    merged = regret_df.merge(rho_df, on=["size", "objective"])
    merged["size"] = pd.Categorical(merged["size"], categories=SIZE_KEYS, ordered=True)
    return merged.sort_values(["objective", "size"]).reset_index(drop=True)


def joint_search_size_distribution() -> pd.DataFrame:
    bf = pd.read_csv(HERE / "qwen3_combined_dpo" / "results" / "best_found.csv")
    counts = bf.groupby(["search_objective", "size"]).size().unstack(fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct = pct.reindex(columns=[s for s in SIZE_KEYS if s in pct.columns])
    return pct.round(1)


def main() -> None:
    fid = fidelity_triangulation()
    fid.to_csv(OUT_DIR / "fidelity_triangulation.csv", index=False)
    print("--- Fidelity triangulation: step 400 (~20%) proxy vs. final checkpoint ---")
    print(fid.to_string(index=False))

    joint = joint_search_size_distribution()
    joint.to_csv(OUT_DIR / "joint_search_size_distribution.csv")
    print("\n--- Joint size+HP search: % of 25 seeds x 6 optimizers recommending each size ---")
    print(joint.to_string())
    pooled = joint.mean(axis=0)
    print("\nPooled across both search objectives:")
    print(pooled.round(1).to_string())


if __name__ == "__main__":
    main()
