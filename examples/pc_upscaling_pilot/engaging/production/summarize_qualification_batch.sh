#!/usr/bin/env bash
# Summarize submitted qualification chains and completed stage markers.

set -euo pipefail

BATCH_ROOT="${1:-/home/shaowen/orcd/scratch/predict_shaowen/runs/qualification_ccu_20260722_v2_fullgrid}"
MANIFEST="${BATCH_ROOT}/submission_manifest.csv"
[[ -f "${MANIFEST}" ]] || { echo "Missing submission manifest: ${MANIFEST}" >&2; exit 2; }

echo "Qualification batch: ${BATCH_ROOT}"
echo "Submitted cases: $(( $(wc -l < "${MANIFEST}") - 1 ))"
echo

job_ids="$(tail -n +2 "${MANIFEST}" | cut -d, -f8-10 | tr ',' '\n' | paste -sd, -)"
if [[ -n "${job_ids}" ]]; then
    sacct -j "${job_ids}" --format=JobID,JobName%24,State,Elapsed,MaxRSS,ExitCode -X
fi

echo
for stage in replay pc kr; do
    count="$(find "${BATCH_ROOT}/cases" -path "*/status/${stage}.done" -type f 2>/dev/null | wc -l)"
    printf '%-8s completed markers: %s\n' "${stage}" "${count}"
done

echo
echo "Failed-stage status files:"
find "${BATCH_ROOT}/cases" -path '*/diagnostics/*.status' -type f -print0 2>/dev/null |
    xargs -0 grep -l '^exit_code=[^0]' 2>/dev/null || true
