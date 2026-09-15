#!/usr/bin/env bash
#SBATCH --job-name=hpo_bb_surrogate
#SBATCH --account=p_neurasearch
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -p barnard
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/hpo_blackbox/.logs/%x_%j.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/hpo_blackbox/.logs/%x_%j.err
#
# Continuous-space counterpart to run_size.sbatch: build_blackbox_surrogate.py
# (--surrogate knn1/knn3/knn5/gp, log10-space lr/beta over the same cleaned
# grid data) -> run_simulations_surrogate.py -> analyze_results_surrogate.py
# (regret curves only), generous budget only, for exactly one (model size,
# surrogate) pair (set via SIZE / SURROGATE / FAMILY, FAMILY defaulting to
# "qwen3").
#
# Submit once per (size, surrogate), in parallel:
#   sbatch --job-name=hpo_bb_surrogate_0.6b_gp --export=ALL,SIZE=0.6b,SURROGATE=gp run_surrogate.sbatch
#
# --n-procs 16 in run_simulations_surrogate.py matches --cpus-per-task above --
# update both together if this job's CPU allocation changes.

set -uo pipefail

: "${SIZE:?Set SIZE, e.g. sbatch --job-name=hpo_bb_surrogate_0.6b_gp --export=ALL,SIZE=0.6b,SURROGATE=gp run_surrogate.sbatch}"
: "${SURROGATE:?Set SURROGATE to one of knn1/knn3/knn5/gp}"
FAMILY="${FAMILY:-qwen3}"

REPO=/data/horse/ws/hama901h-BFTranslation
cd "${REPO}/hpo_blackbox"

PYTHON="${REPO}/syne-tune/.venv/bin/python"

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

t0=$(date +%s)
echo "${FAMILY}-${SIZE} (surrogate=${SURROGATE}): starting pipeline"

run_step "build_blackbox_surrogate.py --size ${SIZE} --family ${FAMILY} --surrogate ${SURROGATE}" \
    "${PYTHON}" build_blackbox_surrogate.py --size "${SIZE}" --family "${FAMILY}" --surrogate "${SURROGATE}"

run_step "run_simulations_surrogate.py --size ${SIZE} --family ${FAMILY} --surrogate ${SURROGATE}" \
    "${PYTHON}" run_simulations_surrogate.py --size "${SIZE}" --family "${FAMILY}" --surrogate "${SURROGATE}" --n-procs 16

run_step "analyze_results_surrogate.py --size ${SIZE} --family ${FAMILY} --surrogate ${SURROGATE}" \
    "${PYTHON}" analyze_results_surrogate.py --size "${SIZE}" --family "${FAMILY}" --surrogate "${SURROGATE}"

t1=$(date +%s)
echo ""
echo "#################################################################"
echo "# ${FAMILY}-${SIZE} (surrogate=${SURROGATE}) done in $(( (t1 - t0) / 60 )) min, ${STEP_FAILURES} step(s) failed."
echo "#################################################################"

exit 0
