#!/usr/bin/env bash
# Plan or submit a restartable node-bundled checkpoint replay/Pc run.

set -euo pipefail

ACTION="${1:-plan}"
MODE="${2:-pilot}"
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
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-0.005}"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if [[ "${MODE}" == "pilot" ]]; then
    NODE_COUNT="${NODE_COUNT:-1}"
    WORKERS_PER_NODE="${WORKERS_PER_NODE:-12}"
    MAX_GROUPS="${MAX_GROUPS:-12}"
    SELECTION="${SELECTION:-largest}"
    WALLTIME="${WALLTIME:-04:00:00}"
else
    NODE_COUNT="${NODE_COUNT:-6}"
    WORKERS_PER_NODE="${WORKERS_PER_NODE:-12}"
    MAX_GROUPS="${MAX_GROUPS:-0}"
    SELECTION="${SELECTION:-all}"
    WALLTIME="${WALLTIME:-24:00:00}"
fi
LANE_COUNT=$((NODE_COUNT * WORKERS_PER_NODE))
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
[[ -z "$(git -C "${RUNTIME_REPO}" status --porcelain)" ]] || {
    echo "Scientific workflow checkout is dirty: ${RUNTIME_REPO}" >&2; exit 2; }
[[ -z "$(git -C "${PREDICT_CODE_ROOT}" status --porcelain)" ]] || {
    echo "PREDICT physics checkout is dirty: ${PREDICT_CODE_ROOT}" >&2; exit 2; }
[[ -z "$(git -C "${SCHEDULER_REPO}" status --porcelain)" ]] || {
    echo "Scheduler checkout is dirty: ${SCHEDULER_REPO}" >&2; exit 2; }

if [[ "${ACTION}" == "submit" ]]; then
    active="$(squeue -u "${USER}" -h -o '%j|%T' | grep -F "${RUN_ID}" || true)"
    if [[ -n "${active}" ]]; then
        echo "Active jobs already exist for ${RUN_ID}:" >&2
        echo "${active}" >&2
        exit 2
    fi
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE_ID="${BUNDLE_ID:-checkpoint_${MODE}_${timestamp}}"
BUNDLE_STATUS_ROOT="${RUN_ROOT}/checkpoint_scheduler/${BUNDLE_ID}"
LANE_MANIFEST="${BUNDLE_STATUS_ROOT}/checkpoint_lanes.csv"
LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}/checkpoint_node_bundle/${BUNDLE_ID}"
builder_args=(
    --run-root "${RUN_ROOT}"
    --output-root "${BUNDLE_STATUS_ROOT}"
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
        --run-root "${RUN_ROOT}" \
        --output-root "${temporary}/bundle" \
        --lane-count "${LANE_COUNT}" \
        --selection "${SELECTION}"
    )
    if (( MAX_GROUPS > 0 )); then
        plan_args+=(--max-groups "${MAX_GROUPS}")
    fi
    python3 "${SCRIPT_DIR}/build_checkpoint_lane_manifest.py" "${plan_args[@]}"
    cat <<EOF
Node-bundled checkpoint plan
  run_id: ${RUN_ID}
  mode: ${MODE}
  nodes: ${NODE_COUNT}
  workers per node: ${WORKERS_PER_NODE}
  one CPU per worker: yes
  memory budget per worker: ${MEMORY_GIB_PER_WORKER} GiB
  memory per node: ${NODE_MEMORY}
  node-local allowance per worker: ${NODE_LOCAL_GIB_PER_WORKER} GiB
  walltime: ${WALLTIME}
  scheduler commit: ${scheduler_commit}
  scientific workflow commit: ${runtime_commit}
  PREDICT physics commit: ${physics_commit}
EOF
    exit 0
fi

mkdir -p "$(dirname "${BUNDLE_STATUS_ROOT}")" "${LOG_ROOT}"
python3 "${SCRIPT_DIR}/build_checkpoint_lane_manifest.py" "${builder_args[@]}"
actual_lane_count="$(python3 - "${BUNDLE_STATUS_ROOT}/checkpoint_lane_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["lane_count"])
PY
)"
LANE_COUNT="${actual_lane_count}"
NODE_COUNT=$(( (LANE_COUNT + WORKERS_PER_NODE - 1) / WORKERS_PER_NODE ))

submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="rpcb_${RUN_ID}" \
        --time="${WALLTIME}" \
        --nodes="${NODE_COUNT}" \
        --ntasks="${LANE_COUNT}" \
        --ntasks-per-node="${WORKERS_PER_NODE}" \
        --cpus-per-task=1 \
        --mem="${NODE_MEMORY}" \
        --output="${LOG_ROOT}/%x_%j.out" \
        --error="${LOG_ROOT}/%x_%j.err" \
        --export=ALL,SCHEDULER_REPO="${SCHEDULER_REPO}",RUNTIME_REPO="${RUNTIME_REPO}",PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT}",FREEZE_ROOT="${FREEZE_ROOT}",PREDICT_ROOT="${PREDICT_ROOT}",METHOD_CONFIG="${METHOD_CONFIG}",RUN_ROOT="${RUN_ROOT}",CHECKPOINT_MANIFEST_ROOT="${RUN_ROOT}/checkpoint_manifest",COMPACT_OUTPUT_ROOT="${RUN_ROOT}/checkpoint_pc",SCRATCH_ROOT="${SCRATCH_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",LANE_MANIFEST="${LANE_MANIFEST}",BUNDLE_STATUS_ROOT="${BUNDLE_STATUS_ROOT}",BUNDLE_LOG_ROOT="${LOG_ROOT}",LANE_COUNT="${LANE_COUNT}",WORKERS_PER_NODE="${WORKERS_PER_NODE}",NODE_LOCAL_GIB_PER_WORKER="${NODE_LOCAL_GIB_PER_WORKER}" \
        "${SCRIPT_DIR}/run_checkpoint_replay_pc_node_bundle.sh"
)"
JOB_ID="${submission%%;*}"
python3 - \
    "${BUNDLE_STATUS_ROOT}/checkpoint_bundle_submission.json" \
    "${BUNDLE_ID}" "${MODE}" "${JOB_ID}" "${scheduler_commit}" \
    "${runtime_commit}" "${physics_commit}" "${NODE_COUNT}" \
    "${WORKERS_PER_NODE}" "${MEMORY_GIB_PER_WORKER}" "${NODE_MEMORY}" "${WALLTIME}" <<'PY'
import json
import sys
from datetime import datetime, timezone

(
    path, bundle_id, mode, job_id, scheduler_commit, runtime_commit,
    physics_commit, nodes, workers, memory_per_worker, memory, walltime,
) = sys.argv[1:]
record = {
    "schema_version": "checkpoint_node_bundle_submission_v1",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "bundle_id": bundle_id,
    "mode": mode,
    "job_id": job_id,
    "scheduler_commit": scheduler_commit,
    "scientific_workflow_commit": runtime_commit,
    "physics_commit": physics_commit,
    "node_count": int(nodes),
    "workers_per_node": int(workers),
    "memory_gib_per_worker": int(memory_per_worker),
    "memory_per_node": memory,
    "walltime": walltime,
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
PY
echo "Submitted checkpoint node bundle ${BUNDLE_ID}: job ${JOB_ID}"
echo "${JOB_ID}" > "${BUNDLE_STATUS_ROOT}/job_id.txt"
