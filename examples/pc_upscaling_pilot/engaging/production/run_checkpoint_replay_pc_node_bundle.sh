#!/usr/bin/env bash
# Launch deterministic one-core checkpoint lanes inside a shared node allocation.

set -euo pipefail

SCHEDULER_REPO="${SCHEDULER_REPO:?SCHEDULER_REPO is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
LANE_MANIFEST="${LANE_MANIFEST:?LANE_MANIFEST is required}"
BUNDLE_STATUS_ROOT="${BUNDLE_STATUS_ROOT:?BUNDLE_STATUS_ROOT is required}"
BUNDLE_LOG_ROOT="${BUNDLE_LOG_ROOT:?BUNDLE_LOG_ROOT is required}"
LANE_COUNT="${LANE_COUNT:?LANE_COUNT is required}"
WORKERS_PER_NODE="${WORKERS_PER_NODE:?WORKERS_PER_NODE is required}"
NODE_LOCAL_GIB_PER_WORKER="${NODE_LOCAL_GIB_PER_WORKER:-20}"
LANE_RUNNER="${SCHEDULER_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc_lane.sh"
VERIFIER="${SCHEDULER_REPO}/examples/pc_upscaling_pilot/engaging/production/verify_checkpoint_lane_completion.py"
NODE_LOCAL_BUNDLE_ROOT="${NODE_LOCAL_BUNDLE_ROOT:-/tmp/${USER}/predict_shaowen/checkpoint_bundle_${SLURM_JOB_ID}}"
export NODE_LOCAL_BUNDLE_ROOT
export WORKERS_PER_NODE NODE_LOCAL_GIB_PER_WORKER

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if (( LANE_COUNT <= 0 || WORKERS_PER_NODE <= 0 )); then
    echo "Lane and per-node worker counts must be positive." >&2
    exit 2
fi
if (( LANE_COUNT > SLURM_NTASKS )); then
    echo "Lane count ${LANE_COUNT} exceeds allocated tasks ${SLURM_NTASKS}." >&2
    exit 2
fi
mkdir -p "${BUNDLE_STATUS_ROOT}/lanes" "${BUNDLE_LOG_ROOT}/lanes"

echo "bundle_job_id=${SLURM_JOB_ID}"
echo "bundle_nodes=${SLURM_NNODES}"
echo "bundle_tasks=${SLURM_NTASKS}"
echo "lane_count=${LANE_COUNT}"
echo "workers_per_node=${WORKERS_PER_NODE}"
echo "node_local_gib_per_worker=${NODE_LOCAL_GIB_PER_WORKER}"
echo "started_at=$(date --iso-8601=seconds)"

# /tmp is not a schedulable Engaging resource, so fail before work starts if
# any assigned node lacks the conservative bundle-level free-space allowance.
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${SLURM_NNODES}" \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    bash -c '
        set -euo pipefail
        required=$(( WORKERS_PER_NODE * NODE_LOCAL_GIB_PER_WORKER * 1024 * 1024 * 1024 ))
        available=$(df -PB1 /tmp | awk "NR==2 {print \$4}")
        echo "node_local_preflight hostname=$(hostname) available_bytes=${available} required_bytes=${required}"
        if (( available < required )); then
            echo "Insufficient node-local /tmp space on $(hostname)." >&2
            exit 2
        fi
    '

set +e
srun \
    --nodes="${SLURM_NNODES}" \
    --ntasks="${LANE_COUNT}" \
    --ntasks-per-node="${WORKERS_PER_NODE}" \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    --distribution=block:block \
    --kill-on-bad-exit=0 \
    --output="${BUNDLE_LOG_ROOT}/lanes/lane_%t.out" \
    --error="${BUNDLE_LOG_ROOT}/lanes/lane_%t.err" \
    bash "${LANE_RUNNER}"
srun_status="$?"
set -e

set +e
python3 "${VERIFIER}" \
    --run-root "${RUN_ROOT}" \
    --lane-manifest "${LANE_MANIFEST}" \
    --output-json "${BUNDLE_STATUS_ROOT}/checkpoint_bundle_completion.json"
verify_status="$?"
set -e

echo "finished_at=$(date --iso-8601=seconds)"
echo "srun_status=${srun_status}"
echo "verification_status=${verify_status}"
if (( srun_status != 0 || verify_status != 0 )); then
    exit 1
fi
