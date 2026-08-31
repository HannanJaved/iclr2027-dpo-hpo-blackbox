#!/usr/bin/env bash
# Submits the 5 per-size HPO blackbox jobs (run_size.sbatch, one per model
# size, independent and run in parallel) plus one combined cross-size job
# (run_combined.sbatch) that waits for all 5 to finish (afterok dependency).
#
# Usage:
#   bash submit_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SIZES=(0.6b 1.7b 4b 8b 14b)
JOB_IDS=()

for size in "${SIZES[@]}"; do
    jid=$(sbatch --parsable --job-name="hpo_bb_${size}" --export=ALL,SIZE="${size}" run_size.sbatch)
    echo "submitted size ${size}: job ${jid}"
    JOB_IDS+=("${jid}")
done

dep="afterok"
for jid in "${JOB_IDS[@]}"; do
    dep="${dep}:${jid}"
done

combined_jid=$(sbatch --parsable --dependency="${dep}" run_combined.sbatch)
echo "submitted combined study (waits on: ${JOB_IDS[*]}): job ${combined_jid}"
