#!/usr/bin/env bash
# Run a contiguous chunk of case-level dynamic-Kr calculations.

set -euo pipefail

ARRAY_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${ARRAY_INDEX}" || ! "${ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 ARRAY_INDEX (or set SLURM_ARRAY_TASK_ID)" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
CASE_COUNT="${CASE_COUNT:?CASE_COUNT is required}"
CASES_PER_ARRAY_TASK="${CASES_PER_ARRAY_TASK:?CASES_PER_ARRAY_TASK is required}"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_case_dynamic_kr.sh"

start_index=$(( (ARRAY_INDEX - 1) * CASES_PER_ARRAY_TASK + 1 ))
end_index=$(( ARRAY_INDEX * CASES_PER_ARRAY_TASK ))
if (( end_index > CASE_COUNT )); then
    end_index="${CASE_COUNT}"
fi
if (( start_index > CASE_COUNT )); then
    echo "No dynamic-Kr work for array index ${ARRAY_INDEX}."
    exit 0
fi

echo "kr_chunk_array_index=${ARRAY_INDEX}"
echo "kr_chunk_start=${start_index}"
echo "kr_chunk_end=${end_index}"

for (( case_index = start_index; case_index <= end_index; case_index++ )); do
    echo "Starting dynamic-Kr case ${case_index}/${CASE_COUNT}."
    bash "${WORKER}" "${case_index}"
done

echo "Completed dynamic-Kr cases ${start_index}-${end_index}."
