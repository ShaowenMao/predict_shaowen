#!/usr/bin/env bash
# Run one restartable node of a checkpoint replay/Pc Slurm array.

set -euo pipefail

SCHEDULER_REPO="${SCHEDULER_REPO:?SCHEDULER_REPO is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
LANE_MANIFEST="${LANE_MANIFEST:?LANE_MANIFEST is required}"
ARRAY_STATUS_ROOT="${ARRAY_STATUS_ROOT:?ARRAY_STATUS_ROOT is required}"
ARRAY_LOG_ROOT="${ARRAY_LOG_ROOT:?ARRAY_LOG_ROOT is required}"
LANE_COUNT="${LANE_COUNT:?LANE_COUNT is required}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:?WORKERS_PER_NODE is required}"
NODE_LOCAL_GIB_PER_WORKER="${NODE_LOCAL_GIB_PER_WORKER:-20}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:?This runner requires a Slurm array task}"
LANE_RUNNER="${SCHEDULER_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc_lane.sh"
VERIFIER="${SCHEDULER_REPO}/examples/pc_upscaling_pilot/engaging/production/verify_checkpoint_lane_completion.py"
NODE_LOCAL_BUNDLE_ROOT="${NODE_LOCAL_BUNDLE_ROOT:-/tmp/${USER}/predict_shaowen/checkpoint_array_${SLURM_ARRAY_JOB_ID}_${ARRAY_TASK_ID}}"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if (( LANE_COUNT <= 0 || WORKERS_PER_NODE <= 0 || ARRAY_TASK_ID <= 0 )); then
    echo "Lane count, workers per node, and array task ID must be positive." >&2
    exit 2
fi

lane_start=$(( (ARRAY_TASK_ID - 1) * WORKERS_PER_NODE + 1 ))
lane_end=$(( ARRAY_TASK_ID * WORKERS_PER_NODE ))
if (( lane_end > LANE_COUNT )); then
    lane_end="${LANE_COUNT}"
fi
if (( lane_start > lane_end )); then
    echo "Array task ${ARRAY_TASK_ID} has no assigned lanes." >&2
    exit 2
fi
local_lane_count=$(( lane_end - lane_start + 1 ))
task_status_root="${ARRAY_STATUS_ROOT}/array_tasks/task_${ARRAY_TASK_ID}"
task_log_root="${ARRAY_LOG_ROOT}/array_tasks/task_${ARRAY_TASK_ID}"

export NODE_LOCAL_BUNDLE_ROOT
export WORKERS_PER_NODE NODE_LOCAL_GIB_PER_WORKER
export LANE_RUNNER LANE_START="${lane_start}"
export BUNDLE_STATUS_ROOT="${ARRAY_STATUS_ROOT}"
mkdir -p "${ARRAY_STATUS_ROOT}/lanes" "${task_status_root}" "${task_log_root}"

echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
echo "array_task_id=${ARRAY_TASK_ID}"
echo "hostname=$(hostname)"
echo "lane_start=${lane_start}"
echo "lane_end=${lane_end}"
echo "local_lane_count=${local_lane_count}"
echo "node_local_gib_per_worker=${NODE_LOCAL_GIB_PER_WORKER}"
echo "started_at=$(date --iso-8601=seconds)"

# /tmp is not schedulable on Engaging, so check the conservative allowance
# before any replay work starts on this node.
required=$(( local_lane_count * NODE_LOCAL_GIB_PER_WORKER * 1024 * 1024 * 1024 ))
available=$(df -PB1 /tmp | awk 'NR==2 {print $4}')
echo "node_local_preflight hostname=$(hostname) available_bytes=${available} required_bytes=${required}"
if (( available < required )); then
    echo "Insufficient node-local /tmp space on $(hostname)." >&2
    exit 2
fi

set +e
srun \
    --nodes=1 \
    --ntasks="${local_lane_count}" \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    --kill-on-bad-exit=0 \
    --output="${task_log_root}/lane_%t.out" \
    --error="${task_log_root}/lane_%t.err" \
    bash -c 'lane_id=$(( LANE_START + SLURM_PROCID )); exec bash "${LANE_RUNNER}" "${lane_id}"'
srun_status="$?"
set -e

set +e
python3 "${VERIFIER}" \
    --run-root "${RUN_ROOT}" \
    --lane-manifest "${LANE_MANIFEST}" \
    --minimum-lane-id "${lane_start}" \
    --maximum-lane-id "${lane_end}" \
    --output-json "${task_status_root}/completion.json"
verify_status="$?"
set -e

echo "finished_at=$(date --iso-8601=seconds)"
echo "srun_status=${srun_status}"
echo "verification_status=${verify_status}"
if (( srun_status != 0 || verify_status != 0 )); then
    exit 1
fi
