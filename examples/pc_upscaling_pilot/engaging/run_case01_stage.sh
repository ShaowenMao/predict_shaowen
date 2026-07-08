#!/usr/bin/env bash
# Run one case01 workflow stage on Engaging.

set -euo pipefail

STAGE="${1:-}"
if [[ -z "${STAGE}" ]]; then
    echo "Usage: $0 replay|pc|kr" >&2
    exit 2
fi

SCRIPT_DIR="${ENGAGING_WORKFLOW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/env_case01.sh"

case "${STAGE}" in
    replay|pc|kr) ;;
    *)
        echo "Unknown stage: ${STAGE}" >&2
        exit 2
        ;;
esac

export UPSCALING_ROOT="${UPSCALING_ROOT:-${RUN_ROOT}/tmp/upscaling_${STAGE}_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "${UPSCALING_ROOT}"
mkdir -p "${RUN_ROOT}/diagnostics"

cd "${REPO_ROOT}/examples/pc_upscaling_pilot"

JOB_ID="${SLURM_JOB_ID:-manual}"
DIAG_PREFIX="${RUN_ROOT}/diagnostics/stage_${STAGE}_${JOB_ID}"
START_EPOCH="$(date +%s)"

finish_stage() {
    local status="$?"
    local end_epoch
    end_epoch="$(date +%s)"
    {
        echo "stage=${STAGE}"
        echo "job_id=${JOB_ID}"
        echo "run_id=${RUN_ID}"
        echo "run_mode=${RUN_MODE}"
        echo "exit_code=${status}"
        echo "start_epoch=${START_EPOCH}"
        echo "end_epoch=${end_epoch}"
        echo "elapsed_seconds=$((end_epoch - START_EPOCH))"
        echo "finished_at=$(date --iso-8601=seconds)"
    } > "${DIAG_PREFIX}.status"
    if [[ "${status}" -eq 0 ]]; then
        touch "${RUN_ROOT}/status/${STAGE}.done"
    fi
}
trap finish_stage EXIT

{
    echo "stage=${STAGE}"
    echo "job_id=${JOB_ID}"
    echo "run_id=${RUN_ID}"
    echo "run_mode=${RUN_MODE}"
    echo "hostname=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "working_directory=$(pwd)"
    echo "slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-}"
    echo "slurm_mem_per_node=${SLURM_MEM_PER_NODE:-}"
    echo "slurm_job_partition=${SLURM_JOB_PARTITION:-}"
    echo "matlab_module=matlab/matlab-2025b"
    echo "mrst_root=${MRST_ROOT}"
    echo "upscaling_zip=${UPSCALING_ZIP}"
    echo "upscaling_root=${UPSCALING_ROOT}"
    { git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/'; } || true
    { git -C "${REPO_ROOT}" status --short 2>/dev/null | sed 's/^/git_status=/'; } || true
} > "${DIAG_PREFIX}.env"

echo "Starting stage=${STAGE} at $(date --iso-8601=seconds)"
echo "Working directory: $(pwd)"

TIME_BIN=""
if [[ -x /usr/bin/time ]]; then
    TIME_BIN="/usr/bin/time"
elif [[ -x /bin/time ]]; then
    TIME_BIN="/bin/time"
fi

run_matlab_stage() {
    local matlab_cmd="$1"
    if [[ -n "${TIME_BIN}" ]]; then
        "${TIME_BIN}" -v -o "${DIAG_PREFIX}.time" \
            matlab -batch "${matlab_cmd}"
    else
        echo "External GNU time not found; running without resource timing file." >&2
        matlab -batch "${matlab_cmd}"
    fi
}

case "${STAGE}" in
    replay)
        run_matlab_stage "prepare_full87_replay_median_examples"
        ;;
    pc)
        run_matlab_stage "run_pc_upscaling_ip_median_examples_full87"
        ;;
    kr)
        run_matlab_stage "run_kr_upscaling_dyn_median_examples_full87"
        ;;
esac

echo "Finished stage=${STAGE} at $(date --iso-8601=seconds)"
