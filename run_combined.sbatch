#!/usr/bin/env bash
#SBATCH --job-name=hpo_bb_combined
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
# Combined cross-size DPO-AO HPO study: build_combined_blackbox.py, its
# simulations/analyses at both budget tags, scale-uncertainty, transfer
# analyses, pinned-fidelity summary, and the converging-evidence summary.
#
# Reads every size's already-built blackbox from disk, so this MUST run
# after all 5 per-size run_size.sbatch jobs have completed -- submit with a
# dependency, e.g.:
#   sbatch --dependency=afterok:<jid1>:<jid2>:<jid3>:<jid4>:<jid5> run_combined.sbatch
# (submit_all.sh does this automatically). It is unaffected by the 8B
# special-case in run_size.sbatch: COMBINED_LR_VALUES/COMBINED_BETA_VALUES
# (common.py) restrict the combined grid to the shared 3x3 subgrid every
# size has always had, so the new 8B/14B sweep points never enter it.

set -uo pipefail

REPO=/data/horse/ws/hama901h-BFTranslation
cd "${REPO}/hpo_blackbox"

PYTHON="${REPO}/syne-tune/.venv/bin/python"
BUDGET_TAGS=("generous" "tight")
TRANSFER_OBJECTIVES=("Z-All" "Z-Dynamic" "Z-Static")

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

run_step "build_combined_blackbox.py" \
    "${PYTHON}" build_combined_blackbox.py

for budget in "${BUDGET_TAGS[@]}"; do
    run_step "run_combined_simulations.py --budget-tag ${budget}" \
        "${PYTHON}" run_combined_simulations.py --budget-tag "${budget}"

    run_step "analyze_combined.py --budget-tag ${budget}" \
        "${PYTHON}" analyze_combined.py --budget-tag "${budget}"

    run_step "analyze_cross_size.py --budget-tag ${budget}" \
        "${PYTHON}" analyze_cross_size.py --budget-tag "${budget}"
done

run_step "analyze_scale_uncertainty.py" \
    "${PYTHON}" analyze_scale_uncertainty.py

for objective in "${TRANSFER_OBJECTIVES[@]}"; do
    run_step "analyze_transfer_across_fidelity.py --objective ${objective}" \
        "${PYTHON}" analyze_transfer_across_fidelity.py --objective "${objective}"

    run_step "analyze_transfer_across_scale.py --objective ${objective}" \
        "${PYTHON}" analyze_transfer_across_scale.py --objective "${objective}"
done

run_step "analyze_pinned_fidelity.py" \
    "${PYTHON}" analyze_pinned_fidelity.py

run_step "summarize_converging_evidence.py" \
    "${PYTHON}" summarize_converging_evidence.py

t1=$(date +%s)
echo ""
echo "#################################################################"
echo "# Combined study done in $(( (t1 - t0) / 60 )) min, ${STEP_FAILURES} step(s) failed."
echo "#################################################################"

exit 0
