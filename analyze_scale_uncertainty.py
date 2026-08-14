#!/usr/bin/env python3
"""
analyze_scale_uncertainty.py

RQ2 ("how does the DPO hyperparameter optimum move with scale?") has so far
been answered with point estimates only: the argmax (lr, beta) of Z-All (or
Z-Dynamic/Z-Static) on each size's final-fidelity grid. A 3x3 grid has zero
residual degrees of freedom to estimate its own fit uncertainty from, so this
script adds uncertainty from two external noise sources.

1. Benchmark-composition bootstrap (primary; cheap, many draws). Z-Dynamic
   and Z-Static are equal-weighted averages over (up to) 4 and 7 benchmarks
   respectively -- an arbitrary choice of weights. This resamples-with-
   replacement WHICH benchmarks contribute to each composite (bootstrapping
   the benchmark list, not the data), recomputes the composite and its
   argmax config each time, and asks: if a different combination of the same
   benchmarks had been used to build the composite, would the same grid cell
   still win? Uses the per-benchmark z-scores already computed by
   analyze_results.load_ground_truth_final (identical formula to
   build_blackbox.add_composite_zscores, just kept per-benchmark instead of
   pre-averaged).

2. ELO judge-noise bootstrap (secondary corroboration; few draws). Each
   config's ELO already has ~10 bootstrap replicates from its own pairwise
   tournament (evaluation_results/openjury-elo/.../elo_bootstrap_ratings.json,
   same file qwen3_iclr_common.build_elo_lookup reads, but keeping the full
   list here instead of collapsing to a mean). Substituting each replicate in
   turn for the point-estimate ELO and recomputing Z-Dynamic/Z-All's argmax
   tests genuine evaluation-noise sensitivity -- but only ~10 draws per
   config, so treat this as a coarse sanity check, not the headline number.

Both bootstraps are run on the shared 3x3 (lr, beta) subgrid only (see
analyze_transfer_across_scale.shared_grid_final), so the "optimum vs. scale"
trend is comparable across all 5 sizes including 4B.

Produces (under figure/analyze_scale_uncertainty/):
  - optimum_lr_vs_scale.png    : log10(LR*) vs log10(size), point = point-
    estimate argmax, errorbar = 10th-90th percentile of the benchmark-
    bootstrap argmax LR, one line per objective (Z-All/Z-Dynamic/Z-Static)
  - optimum_beta_vs_scale.png  : same for beta
  - argmax_stability.png       : P(bootstrap argmax == point-estimate argmax)
    per size x objective -- how often the reported "optimal cell" survives
    resampling
  - scale_uncertainty_table.csv

Usage:
    python analyze_scale_uncertainty.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_transfer_across_scale import SIZE_PARAMS_B, shared_grid_final
from build_blackbox import DYNAMIC_BENCHMARKS, STATIC_BENCHMARKS
from common import HERE, SIZE_KEYS

import sys
sys.path.insert(0, str(HERE.parent / "plot_scripts" / "ICLR"))
import qwen3_iclr_common as iclr  # noqa: E402

OUT_DIR = HERE / "figure" / "analyze_scale_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTIVES = {"Z-All": STATIC_BENCHMARKS + DYNAMIC_BENCHMARKS, "Z-Dynamic": DYNAMIC_BENCHMARKS, "Z-Static": STATIC_BENCHMARKS}
RNG_SEED = 0


def usable_benchmarks_present(df: pd.DataFrame, bench_list: list[str]) -> list[str]:
    return [b for b in bench_list if f"{b}__z" in df.columns and df[f"{b}__z"].notna().all()]


def point_estimate_argmax(df: pd.DataFrame, objective: str) -> tuple[float, float]:
    row = df.loc[df[objective].idxmax()]
    return float(row["lr"]), float(row["beta"])


def benchmark_bootstrap(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Returns, per objective, a DataFrame of n_boot rows with the argmax
    (lr, beta) chosen on that bootstrap replicate's resampled composite."""
    out = {}
    for objective, bench_list in OBJECTIVES.items():
        present = usable_benchmarks_present(df, bench_list)
        if not present:
            continue
        z_matrix = df[[f"{b}__z" for b in present]].to_numpy()  # (n_configs, n_bench)
        draws = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(present), size=len(present))
            boot_composite = z_matrix[:, idx].mean(axis=1)
            winner = df.iloc[int(np.argmax(boot_composite))]
            draws.append({"lr": float(winner["lr"]), "beta": float(winner["beta"])})
        out[objective] = pd.DataFrame(draws)
    return out


def build_elo_bootstrap_lookup(root: Path) -> dict[str, list[float]]:
    """Like qwen3_iclr_common.build_elo_lookup but keeps the raw replicate
    list instead of collapsing to a mean."""
    lookup: dict[str, list[float]] = {}
    if not root.exists():
        return lookup
    for summary_file in root.rglob("summary.json"):
        elo_file = summary_file.with_name("elo_bootstrap_ratings.json")
        if not elo_file.exists():
            continue
        try:
            summary = json.loads(summary_file.read_text())
            bootstrap = json.loads(elo_file.read_text())
        except Exception:
            continue
        model_name = str(summary.get("model", ""))
        if not model_name.startswith("VLLM/"):
            continue
        model_value = model_name[5:]
        vals = [row[model_name] for row in bootstrap if isinstance(row, dict) and model_name in row]
        if vals:
            lookup[model_value] = [float(v) for v in vals]
    return lookup


def elo_judge_bootstrap(size_key: str, df: pd.DataFrame) -> pd.DataFrame | None:
    """Secondary corroboration: substitute each of the ~10 ELO bootstrap
    replicates in place of the point-estimate ELO, recompute Z-All's argmax."""
    elo_boot = build_elo_bootstrap_lookup(iclr.ELO_QWEN_DIR)
    final_models = iclr.load_dpo_final_models(size_key)

    per_config_replicates: dict[tuple[float, float], list[float]] = {}
    for model_name in final_models:
        lr, beta = iclr.parse_dpo_lr_beta(model_name)
        if lr is None:
            continue
        key = (float(lr), float(beta))
        if key not in set(zip(df["lr"], df["beta"])):
            continue  # not on the shared subgrid
        path = iclr.model_path(model_name)
        for candidate in iclr.model_path_aliases(path):
            if candidate in elo_boot:
                per_config_replicates[key] = elo_boot[candidate]
                break

    n_shared = len(set(zip(df["lr"], df["beta"])))
    if len(per_config_replicates) < n_shared:
        missing = n_shared - len(per_config_replicates)
        warnings.warn(f"{size_key}: ELO-bootstrap lookup missing for {missing}/{n_shared} shared configs, skipping")
        return None

    n_reps = min(len(v) for v in per_config_replicates.values())
    if n_reps < 2:
        warnings.warn(f"{size_key}: fewer than 2 ELO-bootstrap replicates available, skipping")
        return None

    other_dynamic = [b for b in DYNAMIC_BENCHMARKS if b != "ELO" and f"{b}__z" in df.columns]
    static_present = usable_benchmarks_present(df, STATIC_BENCHMARKS)
    draws = []
    for r in range(n_reps):
        elo_raw = np.array([per_config_replicates[(row["lr"], row["beta"])][r] for _, row in df.iterrows()])
        elo_z = (elo_raw - elo_raw.mean()) / elo_raw.std(ddof=0) if elo_raw.std(ddof=0) > 1e-12 else np.zeros_like(elo_raw)
        z_dynamic_boot = np.vstack([elo_z] + [df[f"{b}__z"].to_numpy() for b in other_dynamic]).mean(axis=0)
        z_all_boot = np.vstack(
            [elo_z] + [df[f"{b}__z"].to_numpy() for b in other_dynamic] + [df[f"{b}__z"].to_numpy() for b in static_present]
        ).mean(axis=0)
        winner_dyn = df.iloc[int(np.argmax(z_dynamic_boot))]
        winner_all = df.iloc[int(np.argmax(z_all_boot))]
        draws.append({
            "replicate": r,
            "zdyn_lr": float(winner_dyn["lr"]), "zdyn_beta": float(winner_dyn["beta"]),
            "zall_lr": float(winner_all["lr"]), "zall_beta": float(winner_all["beta"]),
        })
    return pd.DataFrame(draws)


def run(n_boot: int) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for size_key in SIZE_KEYS:
        df = shared_grid_final(size_key)
        boot = benchmark_bootstrap(df, n_boot, rng)
        elo_boot = elo_judge_bootstrap(size_key, df)

        for objective in OBJECTIVES:
            if objective not in boot:
                continue
            point_lr, point_beta = point_estimate_argmax(df, objective)
            b = boot[objective]
            stability = float(((b["lr"] == point_lr) & (b["beta"] == point_beta)).mean())
            row = {
                "size": size_key, "size_params_b": SIZE_PARAMS_B[size_key], "objective": objective,
                "point_lr": point_lr, "point_beta": point_beta,
                "boot_lr_p10": np.percentile(b["lr"], 10), "boot_lr_p50": np.percentile(b["lr"], 50),
                "boot_lr_p90": np.percentile(b["lr"], 90),
                "boot_beta_p10": np.percentile(b["beta"], 10), "boot_beta_p50": np.percentile(b["beta"], 50),
                "boot_beta_p90": np.percentile(b["beta"], 90),
                "benchmark_boot_stability": stability,
            }
            if elo_boot is not None and objective in ("Z-All", "Z-Dynamic"):
                col_prefix = "zall" if objective == "Z-All" else "zdyn"
                elo_lr = elo_boot[f"{col_prefix}_lr"]
                elo_beta = elo_boot[f"{col_prefix}_beta"]
                row["elo_boot_n_reps"] = len(elo_boot)
                row["elo_boot_stability"] = float(((elo_lr == point_lr) & (elo_beta == point_beta)).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_optimum_vs_scale(table: pd.DataFrame, param: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = {"Z-All": "#4C72B0", "Z-Dynamic": "#DD8452", "Z-Static": "#55A868"}
    for objective, sub in table.groupby("objective"):
        sub = sub.sort_values("size_params_b")
        x = np.log10(sub["size_params_b"])
        y = np.log10(sub[f"point_{param}"])
        yerr_lo = y - np.log10(sub[f"boot_{param}_p10"].clip(lower=1e-12))
        yerr_hi = np.log10(sub[f"boot_{param}_p90"]) - y
        ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], marker="o", capsize=4, label=objective,
                    color=colors.get(objective, None), linewidth=1.5)
    ax.set_xlabel("log10(model size, B params)")
    ax.set_ylabel(f"log10({'LR*' if param == 'lr' else 'beta*'})")
    ax.set_title(f"Optimal {'learning rate' if param == 'lr' else 'beta'} vs. scale\n"
                 f"(point = argmax on shared 3x3 grid; error bars = 10-90th pctile of\n"
                 f"benchmark-composition bootstrap argmax, {N_BOOT_USED} draws)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"optimum_{param}_vs_scale.png", dpi=150)
    plt.close(fig)


def plot_stability(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    objectives = list(OBJECTIVES.keys())
    width = 0.25
    x = np.arange(len(SIZE_KEYS))
    for i, objective in enumerate(objectives):
        sub = table[table["objective"] == objective].set_index("size").reindex(SIZE_KEYS)
        ax.bar(x + (i - 1) * width, sub["benchmark_boot_stability"], width, label=objective)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-{s}" for s in SIZE_KEYS])
    ax.set_ylabel("P(bootstrap argmax == point-estimate argmax)")
    ax.set_title("How stable is the reported 'optimal cell' under benchmark-composition resampling?")
    ax.axhline(1 / 9, color="grey", linestyle="--", linewidth=1, label="uniform-over-9-cells reference")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "argmax_stability.png", dpi=150)
    plt.close(fig)


N_BOOT_USED = None


def main() -> None:
    global N_BOOT_USED
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    N_BOOT_USED = args.n_boot

    table = run(args.n_boot)
    table.to_csv(OUT_DIR / "scale_uncertainty_table.csv", index=False)
    plot_optimum_vs_scale(table, "lr")
    plot_optimum_vs_scale(table, "beta")
    plot_stability(table)

    print(f"[uncertainty] wrote {len(table)}-row table to {OUT_DIR / 'scale_uncertainty_table.csv'}")
    zall = table[table["objective"] == "Z-All"]
    print("[uncertainty] Z-All argmax stability by size:")
    print(zall[["size", "point_lr", "point_beta", "benchmark_boot_stability"]].to_string(index=False))


if __name__ == "__main__":
    main()
