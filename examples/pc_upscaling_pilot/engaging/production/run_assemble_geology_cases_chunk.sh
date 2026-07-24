#!/usr/bin/env bash
# Assemble a contiguous chunk of geology-level case inputs.

set -euo pipefail

ARRAY_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${ARRAY_INDEX}" || ! "${ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 ARRAY_INDEX (or set SLURM_ARRAY_TASK_ID)" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
GEOLOGY_COUNT="${GEOLOGY_COUNT:?GEOLOGY_COUNT is required}"
GEOLOGIES_PER_ARRAY_TASK="${GEOLOGIES_PER_ARRAY_TASK:?GEOLOGIES_PER_ARRAY_TASK is required}"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_assemble_geology_cases.sh"

start_index=$(( (ARRAY_INDEX - 1) * GEOLOGIES_PER_ARRAY_TASK + 1 ))
end_index=$(( ARRAY_INDEX * GEOLOGIES_PER_ARRAY_TASK ))
if (( end_index > GEOLOGY_COUNT )); then
    end_index="${GEOLOGY_COUNT}"
fi
if (( start_index > GEOLOGY_COUNT )); then
    echo "No assembly work for array index ${ARRAY_INDEX}."
    exit 0
fi

echo "assembly_chunk_array_index=${ARRAY_INDEX}"
echo "assembly_chunk_start=${start_index}"
echo "assembly_chunk_end=${end_index}"

for (( geology_index = start_index; geology_index <= end_index; geology_index++ )); do
    echo "Starting geology assembly ${geology_index}/${GEOLOGY_COUNT}."
    bash "${WORKER}" "${geology_index}"
done

echo "Completed geology assemblies ${start_index}-${end_index}."
