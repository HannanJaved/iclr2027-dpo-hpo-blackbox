"""Shared constants and per-size paths for the Qwen3 DPO multi-fidelity HPO blackbox.

One blackbox per model size (0.6b, 1.7b, 4b, 8b, 14b), never mixed -- HPO
comparisons only make sense within a fixed model size. Use ``get_paths(size_key)``
to resolve where a given size's data/blackbox/results live.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path("/data/horse/ws/hama901h-BFTranslation")
ICLR_DIR = REPO_ROOT / "plot_scripts" / "ICLR"
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ICLR_DIR))

SIZE_KEYS = ["0.6b", "1.7b", "4b", "8b", "14b"]

# All 11 benchmarks tracked by the ICLR pipeline (qwen3_iclr_common.ALL_BENCHMARKS).
BENCHMARKS = [
    "ARC-C", "GPQA", "GSM8K", "HellaSwag", "PIQA", "TruthfulQA", "IFEval",  # static
    "Arena-Hard", "MT-Bench", "AlpacaEval", "ELO",                          # dynamic
]

# Composite objectives computed on top of the raw benchmarks (per-fidelity z-score
# averages, see build_blackbox.py). "higher is better" for all of them, like the
# raw benchmarks.
COMPOSITE_OBJECTIVES = ["Z-Static", "Z-Dynamic", "Z-All"]

# Objective used as the elapsed-time / cost column consumed by the simulator backend.
TIME_OBJECTIVE = "elapsed_time_sec"

ALL_OBJECTIVES = BENCHMARKS + COMPOSITE_OBJECTIVES + [TIME_OBJECTIVE]

FIDELITY_ATTR = "dpo_step"

# Objective metrics we run HPO simulations against. "ELO" and "Z-All" are the two
# headliners (a real preference signal, and a full-grid composite ground truth);
# Arena-Hard / MT-Bench / AlpacaEval are the other pairwise-judge win-rates (to see
# how well optimizing one transfers to the others); IFEval is the odd one out -- a
# static, non-judged instruction-following benchmark, included to see whether
# optimizing a "objective" metric picks different configs than the preference judges.
SEARCH_OBJECTIVES = ["ELO", "Arena-Hard", "MT-Bench", "AlpacaEval", "IFEval", "Z-All"]

# Optimizers compared: RandomSearch (model-free, single-fidelity, the baseline);
# three single-fidelity *model-based* methods that each use a different surrogate
# class (BOTorch = Gaussian process, TPE = kernel density estimate, CQR = conformal
# quantile regression via gradient-boosted trees); and two multi-fidelity methods
# (ASHA = model-free early stopping, BOHB = ASHA's early stopping + TPE's KDE
# model). BOHB vs TPE isolates the effect of multi-fidelity while holding the
# search model fixed; BOTorch/TPE/CQR isolate the effect of surrogate model choice
# while holding the fidelity strategy (single) fixed.
OPTIMIZERS = ["RandomSearch", "BOTorch", "TPE", "CQR", "ASHA", "BOHB"]
OPTIMIZER_COLORS = {
    "RandomSearch": "#4C72B0",
    "BOTorch": "#55A868",
    "TPE": "#8172B2",
    "CQR": "#CCB974",
    "ASHA": "#DD8452",
    "BOHB": "#C44E52",
}

N_SEEDS = 25
N_WORKERS = 2

# Wallclock budget per simulated run, expressed as a multiple of the cost of
# ONE full-fidelity DPO training run for that size. "generous" (the default,
# used everywhere so far) gives single-fidelity methods room to complete ~8
# full trials with 2 workers -- enough to cover most of a 9-16 config grid by
# chance alone, which is why RandomSearch looks deceptively strong there (see
# discussion). "tight" gives just enough budget for 2 workers to each finish
# ONE full trial (4 total workers-worth of runtime across the sweep), which
# is the adversarial regime for non-adaptive search: a blind sampler only
# gets to try 2 of 9-16 configs, while ASHA/BOHB can still try many more
# (cheap, partial) ones in the same window.
BUDGET_REGIMES = {"generous": 3.6, "tight": 1.0}
DEFAULT_BUDGET_TAG = "generous"


def slug(size_key: str) -> str:
    """'1.7b' -> '1p7b', '0.6b' -> '0p6b', '4b' -> '4b'."""
    return size_key.replace(".", "p")


@dataclass(frozen=True)
class Paths:
    size_key: str
    budget_tag: str
    size_dir: Path
    grid_csv: Path
    blackbox_dir: Path
    results_dir: Path
    simulation_raw_csv: Path
    best_found_csv: Path
    figures_dir: Path


def get_paths(size_key: str, budget_tag: str = DEFAULT_BUDGET_TAG) -> Paths:
    if size_key not in SIZE_KEYS:
        raise ValueError(f"size_key must be one of {SIZE_KEYS}, got {size_key!r}")
    if budget_tag not in BUDGET_REGIMES:
        raise ValueError(f"budget_tag must be one of {list(BUDGET_REGIMES)}, got {budget_tag!r}")
    sl = slug(size_key)
    size_dir = HERE / f"qwen3_{sl}_dpo"
    # The default regime keeps its original path (results/) so nothing already
    # computed needs to move; other regimes get their own results_<tag>/ dir.
    results_dirname = "results" if budget_tag == DEFAULT_BUDGET_TAG else f"results_{budget_tag}"
    results_dir = size_dir / results_dirname
    return Paths(
        size_key=size_key,
        budget_tag=budget_tag,
        size_dir=size_dir,
        grid_csv=size_dir / "data" / f"qwen3_{sl}_dpo_grid.csv",
        blackbox_dir=size_dir / "blackbox" / f"qwen3-{size_key}-dpo-ao",
        results_dir=results_dir,
        simulation_raw_csv=results_dir / "simulation_raw.csv",
        best_found_csv=results_dir / "best_found.csv",
        figures_dir=results_dir / "figures",
    )


# --- Combined cross-size experiment -----------------------------------------
# One joint blackbox where "size" is itself a searchable hyperparameter,
# alongside lr and beta, restricted to the 3x3 subgrid shared by every size
# (4B's extra lr=8e-6/beta=0.08 sweep points are dropped for this experiment
# so the config space stays a clean rectangle: 5 sizes x 3 lrs x 3 betas = 45
# configs). Search objectives are ELO (already on one shared cross-size
# tournament scale) and Z-All (recomputed pooling all 45 configs together,
# rather than per-size, so it also reflects absolute cross-size strength).
COMBINED_LR_VALUES = [1e-6, 2e-6, 4e-6]
COMBINED_BETA_VALUES = [0.01, 0.02, 0.04]
COMBINED_SEARCH_OBJECTIVES = ["ELO", "Z-All"]
COMBINED_DIR = HERE / "qwen3_combined_dpo"


@dataclass(frozen=True)
class CombinedPaths:
    budget_tag: str
    combined_dir: Path
    grid_csv: Path
    blackbox_dir: Path
    results_dir: Path
    simulation_raw_csv: Path
    best_found_csv: Path
    figures_dir: Path


def get_combined_paths(budget_tag: str = DEFAULT_BUDGET_TAG) -> CombinedPaths:
    if budget_tag not in BUDGET_REGIMES:
        raise ValueError(f"budget_tag must be one of {list(BUDGET_REGIMES)}, got {budget_tag!r}")
    results_dirname = "results" if budget_tag == DEFAULT_BUDGET_TAG else f"results_{budget_tag}"
    results_dir = COMBINED_DIR / results_dirname
    return CombinedPaths(
        budget_tag=budget_tag,
        combined_dir=COMBINED_DIR,
        grid_csv=COMBINED_DIR / "data" / "qwen3_combined_dpo_grid.csv",
        blackbox_dir=COMBINED_DIR / "blackbox" / "qwen3-combined-dpo-ao",
        results_dir=results_dir,
        simulation_raw_csv=results_dir / "simulation_raw.csv",
        best_found_csv=results_dir / "best_found.csv",
        figures_dir=results_dir / "figures",
    )
