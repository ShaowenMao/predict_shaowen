#!/usr/bin/env bash
# Run a contiguous chunk of checkpoint replay/Pc manifest entries.

set -euo pipefail

ARRAY_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${ARRAY_INDEX}" || ! "${ARRAY_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 ARRAY_INDEX (or set SLURM_ARRAY_TASK_ID)" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
GROUP_COUNT="${GROUP_COUNT:?GROUP_COUNT is required}"
GROUPS_PER_ARRAY_TASK="${GROUPS_PER_ARRAY_TASK:?GROUPS_PER_ARRAY_TASK is required}"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc.sh"

start_index=$(( (ARRAY_INDEX - 1) * GROUPS_PER_ARRAY_TASK + 1 ))
end_index=$(( ARRAY_INDEX * GROUPS_PER_ARRAY_TASK ))
if (( end_index > GROUP_COUNT )); then
    end_index="${GROUP_COUNT}"
fi
if (( start_index > GROUP_COUNT )); then
    echo "No checkpoint work for array index ${ARRAY_INDEX}."
    exit 0
fi

echo "checkpoint_chunk_array_index=${ARRAY_INDEX}"
echo "checkpoint_chunk_start=${start_index}"
echo "checkpoint_chunk_end=${end_index}"

for (( group_index = start_index; group_index <= end_index; group_index++ )); do
    echo "Starting checkpoint group ${group_index}/${GROUP_COUNT}."
    bash "${WORKER}" "${group_index}"
done

echo "Completed checkpoint groups ${start_index}-${end_index}."
