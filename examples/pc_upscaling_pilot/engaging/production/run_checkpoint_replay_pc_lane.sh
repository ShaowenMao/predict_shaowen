#!/usr/bin/env bash
# Process one deterministic checkpoint lane using one CPU and node-local storage.

set -euo pipefail

LANE_ID="${1:-$(( ${SLURM_PROCID:-0} + 1 ))}"
if [[ ! "${LANE_ID}" =~ ^[0-9]+$ || "${LANE_ID}" -le 0 ]]; then
    echo "A positive lane ID is required." >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:?RUNTIME_REPO is required}"
LANE_MANIFEST="${LANE_MANIFEST:?LANE_MANIFEST is required}"
BUNDLE_STATUS_ROOT="${BUNDLE_STATUS_ROOT:?BUNDLE_STATUS_ROOT is required}"
NODE_LOCAL_BUNDLE_ROOT="${NODE_LOCAL_BUNDLE_ROOT:-/tmp/${USER}/predict_shaowen/checkpoint_bundle_${SLURM_JOB_ID:-manual}}"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc.sh"
LANE_ROOT="${NODE_LOCAL_BUNDLE_ROOT}/lane_${LANE_ID}"
export CHECKPOINT_TEMP_ROOT="${LANE_ROOT}/checkpoint"
export MATLAB_SHORT_ROOT="${LANE_ROOT}/matlab"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

mkdir -p "${CHECKPOINT_TEMP_ROOT}" "${MATLAB_SHORT_ROOT}" "${BUNDLE_STATUS_ROOT}/lanes"
started_at="$(date --iso-8601=seconds)"
failure_file="${BUNDLE_STATUS_ROOT}/lanes/lane_${LANE_ID}_failures.tsv"
: > "${failure_file}"

cleanup() {
    local status="$?"
    rm -rf "${LANE_ROOT}"
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

mapfile -t lane_records < <(
    python3 - "${LANE_MANIFEST}" "${LANE_ID}" <<'PY'
import csv
import sys

path, lane_id = sys.argv[1], int(sys.argv[2])
with open(path, newline="", encoding="utf-8-sig") as stream:
    rows = [row for row in csv.DictReader(stream) if int(row["lane_id"]) == lane_id]
for row in sorted(rows, key=lambda item: int(item["lane_position"])):
    print(f"{row['group_index']}\t{row['group_id']}")
PY
)
if (( ${#lane_records[@]} == 0 )); then
    echo "Lane ${LANE_ID} has no checkpoint groups." >&2
    exit 2
fi

attempted=0
completed=0
failed=0
echo "lane_id=${LANE_ID}"
echo "hostname=$(hostname)"
echo "lane_group_count=${#lane_records[@]}"
echo "node_local_lane_root=${LANE_ROOT}"
echo "started_at=${started_at}"

for record in "${lane_records[@]}"; do
    IFS=$'\t' read -r group_index group_id <<< "${record}"
    attempted=$((attempted + 1))
    echo "Starting lane ${LANE_ID} checkpoint group ${group_index} (${group_id})."
    if bash "${WORKER}" "${group_index}"; then
        completed=$((completed + 1))
    else
        worker_status="$?"
        failed=$((failed + 1))
        printf '%s\t%s\t%s\n' "${group_index}" "${group_id}" "${worker_status}" \
            >> "${failure_file}"
        echo "Checkpoint group ${group_index} failed with status ${worker_status}; continuing lane." >&2
    fi
done

finished_at="$(date --iso-8601=seconds)"
python3 - \
    "${BUNDLE_STATUS_ROOT}/lanes/lane_${LANE_ID}.json" \
    "${LANE_ID}" "${started_at}" "${finished_at}" \
    "${attempted}" "${completed}" "${failed}" "$(hostname)" <<'PY'
import json
import sys

path, lane_id, started, finished, attempted, completed, failed, hostname = sys.argv[1:]
record = {
    "schema_version": "checkpoint_lane_execution_v1",
    "lane_id": int(lane_id),
    "hostname": hostname,
    "started_at": started,
    "finished_at": finished,
    "attempted_group_count": int(attempted),
    "completed_group_count": int(completed),
    "failed_group_count": int(failed),
    "passed": int(failed) == 0,
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY

echo "finished_at=${finished_at}"
echo "lane_completed=${completed}"
echo "lane_failed=${failed}"
(( failed == 0 ))
