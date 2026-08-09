#!/usr/bin/env bash
# Plan or submit restartable one-node checkpoint replay/Pc array tasks.

set -euo pipefail

ACTION="${1:-plan}"
MODE="${2:-full}"
if [[ "${ACTION}" != "plan" && "${ACTION}" != "submit" ]]; then
    echo "Usage: $0 plan|submit pilot|full" >&2
    exit 2
fi
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 plan|submit pilot|full" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULER_REPO="${SCHEDULER_REPO:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"
RUNTIME_REPO="${RUNTIME_REPO:?RUNTIME_REPO must be the immutable scientific workflow checkout}"
PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:?PREDICT_CODE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
FREEZE_ROOT="${FREEZE_ROOT:?FREEZE_ROOT is required}"
PREDICT_ROOT="${PREDICT_ROOT:?PREDICT_ROOT is required}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
METHOD_CONFIG="${METHOD_CONFIG:-${FREEZE_ROOT}/config/production_method_config.toml}"
RUN_ID="${RUN_ID:-$(basename "${RUN_ROOT}")}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mit_amf_advanced_cpu}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal}"
MEMORY_GIB_PER_WORKER="${MEMORY_GIB_PER_WORKER:-18}"
NODE_LOCAL_GIB_PER_WORKER="${NODE_LOCAL_GIB_PER_WORKER:-20}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-1.0e-3}"
FALLBACK_JOB_ID="${FALLBACK_JOB_ID:-}"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if [[ "${MODE}" == "pilot" ]]; then
    NODE_TASK_COUNT="${NODE_TASK_COUNT:-1}"
    WORKERS_PER_NODE="${WORKERS_PER_NODE:-12}"
    MAX_GROUPS="${MAX_GROUPS:-12}"
    SELECTION="${SELECTION:-largest}"
    WALLTIME="${WALLTIME:-04:00:00}"
else
    NODE_TASK_COUNT="${NODE_TASK_COUNT:-9}"
    WORKERS_PER_NODE="${WORKERS_PER_NODE:-12}"
    MAX_GROUPS="${MAX_GROUPS:-0}"
    SELECTION="${SELECTION:-all}"
    WALLTIME="${WALLTIME:-36:00:00}"
fi
LANE_COUNT=$(( NODE_TASK_COUNT * WORKERS_PER_NODE ))
NODE_MEMORY="${NODE_MEMORY:-$((WORKERS_PER_NODE * MEMORY_GIB_PER_WORKER))G}"

identity_path="${RUN_ROOT}/phase_run_identity.json"
[[ -f "${identity_path}" ]] || { echo "Missing run identity: ${identity_path}" >&2; exit 2; }
read -r SAMPLING_COMMIT PHYSICS_COMMIT METHOD_CONFIG_SHA256 < <(
    python3 - "${identity_path}" <<'PY'
import json
import sys
identity = json.load(open(sys.argv[1], encoding="utf-8"))
print(
    identity["sampling_code_commit"],
    identity["physics_commit"],
    identity["production_method_config_sha256"],
)
PY
)

runtime_commit="$(git -C "${RUNTIME_REPO}" rev-parse HEAD)"
physics_commit="$(git -C "${PREDICT_CODE_ROOT}" rev-parse HEAD)"
scheduler_commit="$(git -C "${SCHEDULER_REPO}" rev-parse HEAD)"
[[ "${runtime_commit}" == "${SAMPLING_COMMIT}" ]] || {
    echo "Scientific workflow commit mismatch: ${runtime_commit} != ${SAMPLING_COMMIT}" >&2; exit 2; }
[[ "${physics_commit}" == "${PHYSICS_COMMIT}" ]] || {
    echo "PREDICT physics commit mismatch: ${physics_commit} != ${PHYSICS_COMMIT}" >&2; exit 2; }
for checkout in "${RUNTIME_REPO}" "${PREDICT_CODE_ROOT}" "${SCHEDULER_REPO}"; do
    [[ -z "$(git -C "${checkout}" status --porcelain)" ]] || {
        echo "Required checkout is dirty: ${checkout}" >&2; exit 2; }
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
ARRAY_ID="${ARRAY_ID:-checkpoint_${MODE}_node_array_${timestamp}}"
ARRAY_STATUS_ROOT="${RUN_ROOT}/checkpoint_scheduler/${ARRAY_ID}"
LANE_MANIFEST="${ARRAY_STATUS_ROOT}/checkpoint_lanes.csv"
LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}/checkpoint_node_array/${ARRAY_ID}"
builder_args=(
    --run-root "${RUN_ROOT}"
    --output-root "${ARRAY_STATUS_ROOT}"
    --lane-count "${LANE_COUNT}"
    --selection "${SELECTION}"
)
if (( MAX_GROUPS > 0 )); then
    builder_args+=(--max-groups "${MAX_GROUPS}")
fi

if [[ "${ACTION}" == "plan" ]]; then
    temporary="$(mktemp -d)"
    trap 'rm -rf "${temporary}"' EXIT
    plan_args=(
        --run-root "${RUN_ROOT}"
        --output-root "${temporary}/array"
        --lane-count "${LANE_COUNT}"
        --selection "${SELECTION}"
    )
    if (( MAX_GROUPS > 0 )); then
        plan_args+=(--max-groups "${MAX_GROUPS}")
    fi
    python3 "${SCRIPT_DIR}/build_checkpoint_lane_manifest.py" "${plan_args[@]}"
    cat <<EOF
One-node checkpoint array plan
  run_id: ${RUN_ID}
  array tasks: ${NODE_TASK_COUNT}
  workers per task: ${WORKERS_PER_NODE}
  one CPU per worker: yes
  memory budget per worker: ${MEMORY_GIB_PER_WORKER} GiB
  memory per array task: ${NODE_MEMORY}
  walltime: ${WALLTIME}
  fallback job to hold: ${FALLBACK_JOB_ID:-none}
  scheduler commit: ${scheduler_commit}
  scientific workflow commit: ${runtime_commit}
  PREDICT physics commit: ${physics_commit}
EOF
    exit 0
fi

fallback_held=0
array_submitted=0
restore_fallback_on_error() {
    local status="$?"
    if (( status != 0 && fallback_held == 1 && array_submitted == 0 )); then
        echo "Array submission failed; releasing fallback job ${FALLBACK_JOB_ID}." >&2
        scontrol release "${FALLBACK_JOB_ID}" || true
    fi
    exit "${status}"
}
trap restore_fallback_on_error EXIT

if [[ -n "${FALLBACK_JOB_ID}" ]]; then
    fallback_state="$(squeue -h -j "${FALLBACK_JOB_ID}" -o '%T')"
    [[ "${fallback_state}" == "PENDING" ]] || {
        echo "Fallback job ${FALLBACK_JOB_ID} must be pending, observed: ${fallback_state:-missing}" >&2
        exit 2
    }
    scontrol hold "${FALLBACK_JOB_ID}"
    fallback_held=1
    held_reason="$(squeue -h -j "${FALLBACK_JOB_ID}" -o '%R')"
    [[ "${held_reason}" == "JobHeldUser" ]] || {
        echo "Fallback job ${FALLBACK_JOB_ID} was not safely held: ${held_reason}" >&2
        exit 2
    }
fi

mkdir -p "$(dirname "${ARRAY_STATUS_ROOT}")" "${LOG_ROOT}"
python3 "${SCRIPT_DIR}/build_checkpoint_lane_manifest.py" "${builder_args[@]}"
actual_lane_count="$(python3 - "${ARRAY_STATUS_ROOT}/checkpoint_lane_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["lane_count"])
PY
)"
LANE_COUNT="${actual_lane_count}"
NODE_TASK_COUNT=$(( (LANE_COUNT + WORKERS_PER_NODE - 1) / WORKERS_PER_NODE ))

submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="rpca_${RUN_ID}" \
        --time="${WALLTIME}" \
        --nodes=1 \
        --ntasks="${WORKERS_PER_NODE}" \
        --ntasks-per-node="${WORKERS_PER_NODE}" \
        --cpus-per-task=1 \
        --mem="${NODE_MEMORY}" \
        --array="1-${NODE_TASK_COUNT}%${NODE_TASK_COUNT}" \
        --output="${LOG_ROOT}/%x_%A_%a.out" \
        --error="${LOG_ROOT}/%x_%A_%a.err" \
        --export=ALL,SCHEDULER_REPO="${SCHEDULER_REPO}",RUNTIME_REPO="${RUNTIME_REPO}",PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT}",FREEZE_ROOT="${FREEZE_ROOT}",PREDICT_ROOT="${PREDICT_ROOT}",METHOD_CONFIG="${METHOD_CONFIG}",RUN_ROOT="${RUN_ROOT}",CHECKPOINT_MANIFEST_ROOT="${RUN_ROOT}/checkpoint_manifest",COMPACT_OUTPUT_ROOT="${RUN_ROOT}/checkpoint_pc",SCRATCH_ROOT="${SCRATCH_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",LANE_MANIFEST="${LANE_MANIFEST}",ARRAY_STATUS_ROOT="${ARRAY_STATUS_ROOT}",ARRAY_LOG_ROOT="${LOG_ROOT}",LANE_COUNT="${LANE_COUNT}",WORKERS_PER_NODE="${WORKERS_PER_NODE}",NODE_LOCAL_GIB_PER_WORKER="${NODE_LOCAL_GIB_PER_WORKER}" \
        "${SCRIPT_DIR}/run_checkpoint_replay_pc_node_array_task.sh"
)"
JOB_ID="${submission%%;*}"
array_submitted=1

python3 - \
    "${ARRAY_STATUS_ROOT}/checkpoint_node_array_submission.json" \
    "${ARRAY_ID}" "${MODE}" "${JOB_ID}" "${FALLBACK_JOB_ID}" \
    "${scheduler_commit}" "${runtime_commit}" "${physics_commit}" \
    "${NODE_TASK_COUNT}" "${WORKERS_PER_NODE}" "${MEMORY_GIB_PER_WORKER}" \
    "${NODE_MEMORY}" "${WALLTIME}" <<'PY'
import json
import sys
from datetime import datetime, timezone

(
    path, array_id, mode, job_id, fallback_job_id, scheduler_commit,
    runtime_commit, physics_commit, node_tasks, workers,
    memory_per_worker, memory, walltime,
) = sys.argv[1:]
record = {
    "schema_version": "checkpoint_node_array_submission_v1",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "array_id": array_id,
    "mode": mode,
    "job_id": job_id,
    "held_fallback_job_id": fallback_job_id or None,
    "scheduler_commit": scheduler_commit,
    "scientific_workflow_commit": runtime_commit,
    "physics_commit": physics_commit,
    "node_task_count": int(node_tasks),
    "workers_per_node": int(workers),
    "memory_gib_per_worker": int(memory_per_worker),
    "memory_per_node_task": memory,
    "walltime": walltime,
    "fallback_protocol": (
        "After the array completes, cancel the held fallback to release its "
        "afterany validation gate. If the array is abandoned, cancel it before "
        "releasing the fallback."
    ),
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY

trap - EXIT
echo "Submitted one-node checkpoint array ${ARRAY_ID}: job ${JOB_ID}"
echo "Array tasks: ${NODE_TASK_COUNT}; lanes: ${LANE_COUNT}"
if [[ -n "${FALLBACK_JOB_ID}" ]]; then
    echo "Held fallback job: ${FALLBACK_JOB_ID}"
fi
