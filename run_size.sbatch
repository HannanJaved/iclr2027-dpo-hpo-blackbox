#!/usr/bin/env bash
#SBATCH --job-name=hpo_bb_size
#SBATCH --account=p_neurasearch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p barnard
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/hpo_blackbox/.logs/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/hpo_blackbox/.logs/%x_%j.err
#
# Per-size DPO-AO HPO blackbox pipeline: collect -> build -> simulate ->
# analyze, for exactly one model size (set via SIZE). Meant to be submitted
# once per size, in parallel -- see submit_all.sh, or by hand:
#   sbatch --job-name=hpo_bb_0.6b --export=ALL,SIZE=0.6b run_size.sbatch
#
# 8B and 14B special-cases: both DPO sweeps grew from 3x3 (lr1e-6/2e-6/4e-6 x
# beta0.01/0.02/0.04) to 5x3, with 6 new lr4e-7/lr8e-7 configs added.
#
# 8B: as of 2026-08-27 all 6 new configs are fully trained with only scattered
# single-step eval gaps (ARC-C/IFEval missing at a handful of individual
# checkpoints for 2 configs) -- comfortably fillable by build_blackbox.py's
# ffill/bfill, well under its 20%-of-configs drop threshold for any benchmark.
# 8B now rebuilds cleanly on the full 5x3 grid; no filtering needed.
#
# 14B: partially ready as of 2026-08-27. lr4e-7/beta0.04 still hasn't launched
# (no trainer_state.json at all), so lr=4e-7 isn't a full row (only beta0.01/
# 0.02 exist) and can't be included without breaking rectangularity. lr=8e-7,
# however, is now fully trained AND fully evaluated (the IFEval backfill for
# the new configs landed today -- previously it was missing at every fidelity
# for all 5 trained new configs, which would have exceeded the 20%-of-configs
# drop threshold and silently dropped IFEval from 14B's search objectives).
# So for SIZE=14b this script drops only the lr=4e-7 rows from the raw grid
# CSV before build_blackbox.py sees it, giving a clean 4x3=12 config
# rectangle (lr in {1e-6,2e-6,4e-6,8e-7}) with all 11 benchmarks intact.
# No-op for every other size. Once lr4e-7/beta0.04 has trained and evaluated,
# drop the SIZE=14b special case below to fold it back in too.

set -uo pipefail

: "${SIZE:?Set SIZE, e.g. sbatch --job-name=hpo_bb_0.6b --export=ALL,SIZE=0.6b run_size.sbatch}"

REPO=/data/horse/ws/hama901h-BFTranslation
cd "${REPO}/hpo_blackbox"

PYTHON="${REPO}/syne-tune/.venv/bin/python"
BUDGET_TAGS=("generous" "tight")
DROP_LR_VALUES=("4e-07")

STEP_FAILURES=0

run_step() {
    local desc="$1"; shift
    echo ""
    echo "=== [$(date +%T)] START: ${desc} ==="
    if "$@"; then
        echo "=== [$(date +%T)] OK: ${desc} ==="
    else
        local rc=$?
        echo "=== [$(date +%T)] FAILED (exit ${rc}): ${desc} -- continuing with the rest of the pipeline ==="
        STEP_FAILURES=$((STEP_FAILURES + 1))
    fi
}

filter_grid_csv() {
    # Drop rows whose lr matches one of DROP_LR_VALUES from a size's raw grid
    # CSV, in place, before build_blackbox.py reads it.
    local grid_csv="$1"; shift
    "${PYTHON}" - "${grid_csv}" "$@" <<'PYEOF'
import sys
import pandas as pd

grid_csv = sys.argv[1]
drop_lrs = [float(x) for x in sys.argv[2:]]

df = pd.read_csv(grid_csv)
n_before = len(df)
mask = df["lr"].apply(lambda v: any(abs(v - d) < 1e-12 for d in drop_lrs))
df[~mask].to_csv(grid_csv, index=False)
print(f"filter_grid_csv: {grid_csv}: dropped {mask.sum()} of {n_before} rows "
      f"(lr in {drop_lrs}), kept {(~mask).sum()}")
PYEOF
}

t0=$(date +%s)
echo "Size ${SIZE}: starting pipeline"

run_step "collect_data.py --size ${SIZE}" \
    "${PYTHON}" collect_data.py --size "${SIZE}"

if [[ "${SIZE}" == "14b" ]]; then
    slug="${SIZE/./p}"
    grid_csv="qwen3_${slug}_dpo/data/qwen3_${slug}_dpo_grid.csv"
    run_step "filter out lr in {${DROP_LR_VALUES[*]}} from ${grid_csv}" \
        filter_grid_csv "${grid_csv}" "${DROP_LR_VALUES[@]}"
fi

run_step "build_blackbox.py --size ${SIZE}" \
    "${PYTHON}" build_blackbox.py --size "${SIZE}"

for budget in "${BUDGET_TAGS[@]}"; do
    run_step "run_simulations.py --size ${SIZE} --budget-tag ${budget}" \
        "${PYTHON}" run_simulations.py --size "${SIZE}" --budget-tag "${budget}"

    run_step "analyze_results.py --size ${SIZE} --budget-tag ${budget}" \
        "${PYTHON}" analyze_results.py --size "${SIZE}" --budget-tag "${budget}"
done

run_step "run_pinned_fidelity_simulations.py --size ${SIZE}" \
    "${PYTHON}" run_pinned_fidelity_simulations.py --size "${SIZE}"

t1=$(date +%s)
echo ""
echo "#################################################################"
echo "# Size ${SIZE} done in $(( (t1 - t0) / 60 )) min, ${STEP_FAILURES} step(s) failed."
echo "#################################################################"

exit 0
