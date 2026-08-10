#!/usr/bin/env bash
# Resume missing checkpoint groups and rebuild the downstream production chain.

set -euo pipefail

ACTION="${1:-plan}"
if [[ "${ACTION}" != "plan" && "${ACTION}" != "submit" ]]; then
    echo "Usage: $0 plan|submit" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
ORCHESTRATION_COMMIT="${ORCHESTRATION_COMMIT:-}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT:-${FREEZE_ROOT}/code/source}"
PREDICT_ROOT="${PREDICT_ROOT:-${FREEZE_ROOT}/inputs/predict}"
METHOD_CONFIG="${METHOD_CONFIG:-${FREEZE_ROOT}/config/production_method_config.toml}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
RUN_ID="${RUN_ID:-production_all1620_20260724_v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-0.005}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mit_amf_advanced_cpu}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal}"
CHECKPOINT_MAX_CONCURRENT="${CHECKPOINT_MAX_CONCURRENT:-56}"
CHECKPOINT_WALLTIME="${CHECKPOINT_WALLTIME:-24:00:00}"
CHECKPOINT_MEMORY="${CHECKPOINT_MEMORY:-18G}"
NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT:-/tmp/${USER}/predict_shaowen}"
CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${NODE_LOCAL_TMP_ROOT}/checkpoint}"
KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-24}"
SLURM_MAX_SUBMITTED_JOBS="${SLURM_MAX_SUBMITTED_JOBS:-400}"

CHECKPOINT_MANIFEST_ROOT="${RUN_ROOT}/checkpoint_manifest"
CHECKPOINT_OUTPUT_ROOT="${RUN_ROOT}/checkpoint_pc"
GROUPS_CSV="${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv"
LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc_chunk.sh"

[[ -d "${RUN_ROOT}" ]] || {
    echo "Missing production run root: ${RUN_ROOT}" >&2
    exit 2
}
[[ -f "${GROUPS_CSV}" ]] || {
    echo "Missing checkpoint manifest: ${GROUPS_CSV}" >&2
    exit 2
}
if [[ "${ACTION}" == "submit" && -z "${ORCHESTRATION_COMMIT}" ]]; then
    echo "ORCHESTRATION_COMMIT is required for submission provenance." >&2
    exit 2
fi
if [[ "${ACTION}" == "submit" ]]; then
    [[ -d "${RUNTIME_REPO}/.git" ]] || {
        echo "Runtime repository is not a Git worktree: ${RUNTIME_REPO}" >&2
        exit 2
    }
    [[ -d "${PREDICT_CODE_ROOT}/.git" ]] || {
        echo "Frozen PREDICT source is not a Git worktree: ${PREDICT_CODE_ROOT}" >&2
        exit 2
    }
    [[ -d "${PREDICT_ROOT}/data" ]] || {
        echo "Frozen PREDICT data root is missing data/: ${PREDICT_ROOT}" >&2
        exit 2
    }
    [[ -f "${METHOD_CONFIG}" ]] || {
        echo "Missing frozen method configuration: ${METHOD_CONFIG}" >&2
        exit 2
    }
    actual_runtime_commit="$(git -C "${RUNTIME_REPO}" rev-parse HEAD)"
    [[ "${actual_runtime_commit}" == "${ORCHESTRATION_COMMIT}" ]] || {
        echo "Runtime commit mismatch: ${actual_runtime_commit} != ${ORCHESTRATION_COMMIT}" >&2
        exit 2
    }
    [[ -z "$(git -C "${RUNTIME_REPO}" status --porcelain)" ]] || {
        echo "Runtime repository has uncommitted changes: ${RUNTIME_REPO}" >&2
        exit 2
    }
    actual_physics_commit="$(git -C "${PREDICT_CODE_ROOT}" rev-parse HEAD)"
    [[ "${actual_physics_commit}" == "${PHYSICS_COMMIT}" ]] || {
        echo "PREDICT physics commit mismatch: ${actual_physics_commit} != ${PHYSICS_COMMIT}" >&2
        exit 2
    }
    [[ -z "$(git -C "${PREDICT_CODE_ROOT}" status --porcelain)" ]] || {
        echo "Frozen PREDICT repository has uncommitted changes: ${PREDICT_CODE_ROOT}" >&2
        exit 2
    }
    actual_method_hash="$(sha256sum "${METHOD_CONFIG}" | awk '{print $1}')"
    [[ "${actual_method_hash}" == "${METHOD_CONFIG_SHA256}" ]] || {
        echo "Method-config SHA mismatch: ${actual_method_hash} != ${METHOD_CONFIG_SHA256}" >&2
        exit 2
    }
fi

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

mapfile -t missing_indices < <(
    python3 - "${GROUPS_CSV}" "${CHECKPOINT_OUTPUT_ROOT}" \
        "${PHYSICS_COMMIT}" "${METHOD_CONFIG_SHA256}" \
        "${REPLAY_TOLERANCE_LOG10}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

groups_csv = Path(sys.argv[1])
output_root = Path(sys.argv[2])
physics_commit = sys.argv[3]
method_config_sha256 = sys.argv[4]
maximum_tolerance = float(sys.argv[5])
with groups_csv.open(newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))

for row in rows:
    marker = output_root / row["group_id"] / "checkpoint.done.json"
    if not marker.is_file():
        print(row["group_index"])
        continue
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(row["group_index"])
        continue
    expected = {
        "status": "complete",
        "group_id": row["group_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "physics_commit": physics_commit,
        "method_config_sha256": method_config_sha256,
        "task_count": int(row["task_count"]),
    }
    if any(data.get(key) != value for key, value in expected.items()):
        print(row["group_index"])
        continue
    try:
        recorded_tolerance = float(
            data.get("replay_tolerance_log10", math.nan)
        )
        maximum_difference = float(
            data.get("max_replay_abs_log10_difference", math.nan)
        )
    except (TypeError, ValueError):
        print(row["group_index"])
        continue
    if (
        not math.isfinite(recorded_tolerance)
        or recorded_tolerance <= 0.0
        or recorded_tolerance > maximum_tolerance + 1.0e-12
        or not math.isfinite(maximum_difference)
        or maximum_difference < 0.0
        or maximum_difference > recorded_tolerance + 1.0e-12
    ):
        print(row["group_index"])
PY
)
MISSING_COUNT="${#missing_indices[@]}"
if (( MISSING_COUNT == 0 )); then
    echo "All checkpoint groups already have completion markers."
    export RUNTIME_REPO FREEZE_ROOT SCRATCH_ROOT RUN_ID RUN_ROOT
    bash "${SCRIPT_DIR}/submit_geology_stratigraphy_package.sh" "${ACTION}"
    exit 0
fi

GROUP_COUNT="$(
    python3 - "${GROUPS_CSV}" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    print(sum(1 for _ in csv.DictReader(stream)))
PY
)"
MISSING_ARRAY_SPEC="$(IFS=,; echo "${missing_indices[*]}")"
ASSEMBLY_ARRAY_TASK_COUNT=27
KR_ARRAY_TASK_COUNT=162
TOTAL_SUBMITTED_JOB_ELEMENTS=$(( \
    MISSING_COUNT + ASSEMBLY_ARRAY_TASK_COUNT + KR_ARRAY_TASK_COUNT + 3 \
))
if (( TOTAL_SUBMITTED_JOB_ELEMENTS > SLURM_MAX_SUBMITTED_JOBS )); then
    echo "Continuation needs ${TOTAL_SUBMITTED_JOB_ELEMENTS} job elements; limit is ${SLURM_MAX_SUBMITTED_JOBS}." >&2
    exit 2
fi

active_jobs="$(
    squeue -u "${USER}" -h -o "%j|%T" \
        | grep -F "${RUN_ID}" \
        || true
)"
if [[ -n "${active_jobs}" ]]; then
    echo "Active jobs already exist for ${RUN_ID}:" >&2
    echo "${active_jobs}" >&2
    exit 2
fi

cat <<EOF
Production continuation plan
  run_id: ${RUN_ID}
  valid completed checkpoints: $((GROUP_COUNT - MISSING_COUNT))/${GROUP_COUNT}
  missing checkpoint groups: ${MISSING_COUNT}
  one missing group per array task
  checkpoint walltime: ${CHECKPOINT_WALLTIME}
  checkpoint concurrency: ${CHECKPOINT_MAX_CONCURRENT}
  checkpoint temporary root: ${CHECKPOINT_TEMP_ROOT}
  orchestration commit: ${ORCHESTRATION_COMMIT:-not checked in plan mode}
  PREDICT physics commit: ${PHYSICS_COMMIT}
  frozen PREDICT data: ${PREDICT_ROOT}
  method config SHA-256: ${METHOD_CONFIG_SHA256}
  replay tolerance ceiling: ${REPLAY_TOLERANCE_LOG10}
  downstream assembly tasks: ${ASSEMBLY_ARRAY_TASK_COUNT}
  downstream dynamic-Kr tasks: ${KR_ARRAY_TASK_COUNT}
  downstream geology-stratigraphy package jobs: 1
  total submitted job elements: ${TOTAL_SUBMITTED_JOB_ELEMENTS}/${SLURM_MAX_SUBMITTED_JOBS}
EOF

if [[ "${ACTION}" == "plan" ]]; then
    exit 0
fi

mkdir -p \
    "${LOG_ROOT}/checkpoint_pc_continuation" \
    "${LOG_ROOT}/checkpoint_gate_continuation" \
    "${LOG_ROOT}/final_gate_continuation"

checkpoint_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="rpc_${RUN_ID}" \
        --time="${CHECKPOINT_WALLTIME}" \
        --cpus-per-task=1 \
        --mem="${CHECKPOINT_MEMORY}" \
        --array="${MISSING_ARRAY_SPEC}%${CHECKPOINT_MAX_CONCURRENT}" \
        --output="${LOG_ROOT}/checkpoint_pc_continuation/%x_%A_%a.out" \
        --error="${LOG_ROOT}/checkpoint_pc_continuation/%x_%A_%a.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",PREDICT_CODE_ROOT="${PREDICT_CODE_ROOT}",PREDICT_ROOT="${PREDICT_ROOT}",FREEZE_ROOT="${FREEZE_ROOT}",METHOD_CONFIG="${METHOD_CONFIG}",CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT}",COMPACT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT}",CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",GROUP_COUNT="${GROUP_COUNT}",GROUPS_PER_ARRAY_TASK=1 \
        "${WORKER}"
)"
CHECKPOINT_ARRAY_JOB_ID="${checkpoint_submission%%;*}"
echo "${CHECKPOINT_ARRAY_JOB_ID}" > "${RUN_ROOT}/checkpoint_array_job_id.txt"
echo "${CHECKPOINT_ARRAY_JOB_ID}" > "${RUN_ROOT}/checkpoint_continuation_job_id.txt"

checkpoint_gate_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="gate_${RUN_ID}" \
        --time="${CHECKPOINT_GATE_WALLTIME:-04:00:00}" \
        --cpus-per-task=1 \
        --mem="${CHECKPOINT_GATE_MEMORY:-8G}" \
        --dependency="afterany:${CHECKPOINT_ARRAY_JOB_ID}" \
        --output="${LOG_ROOT}/checkpoint_gate_continuation/%x_%j.out" \
        --error="${LOG_ROOT}/checkpoint_gate_continuation/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",DEFAULT_REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}" \
        "${SCRIPT_DIR}/run_checkpoint_completion_gate.sh"
)"
CHECKPOINT_GATE_JOB_ID="${checkpoint_gate_submission%%;*}"

export RUNTIME_REPO FREEZE_ROOT SCRATCH_ROOT RUN_ID RUN_ROOT
export CHECKPOINT_JOB_ID="${CHECKPOINT_GATE_JOB_ID}"
export PHYSICS_COMMIT METHOD_CONFIG_SHA256 REPLAY_TOLERANCE_LOG10
export KR_MAX_CONCURRENT
export GEOLOGIES_PER_ARRAY_TASK=6
export CASES_PER_ARRAY_TASK=10
bash "${SCRIPT_DIR}/submit_case_assembly_kr.sh" full
ASSEMBLY_JOB_ID="$(<"${RUN_ROOT}/assembly_array_job_id.txt")"
KR_JOB_ID="$(<"${RUN_ROOT}/kr_array_job_id.txt")"

final_gate_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="qa_${RUN_ID}" \
        --time="${FINAL_GATE_WALLTIME:-08:00:00}" \
        --cpus-per-task=1 \
        --mem="${FINAL_GATE_MEMORY:-8G}" \
        --dependency="afterany:${KR_JOB_ID}" \
        --output="${LOG_ROOT}/final_gate_continuation/%x_%j.out" \
        --error="${LOG_ROOT}/final_gate_continuation/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${REPLAY_TOLERANCE_LOG10}" \
        "${SCRIPT_DIR}/run_case_completion_gate.sh"
)"
FINAL_GATE_JOB_ID="${final_gate_submission%%;*}"

export DEPENDENCY_JOB_ID="${FINAL_GATE_JOB_ID}"
bash "${SCRIPT_DIR}/submit_geology_stratigraphy_package.sh" submit
STRATIGRAPHY_JOB_ID="$(<"${RUN_ROOT}/geology_stratigraphy_job_id.txt")"
unset DEPENDENCY_JOB_ID

python3 - \
    "${RUN_ROOT}/production_continuation_manifest.json" \
    "${RUN_ID}" \
    "${ORCHESTRATION_COMMIT}" \
    "${PHYSICS_COMMIT}" \
    "${METHOD_CONFIG}" \
    "${METHOD_CONFIG_SHA256}" \
    "${REPLAY_TOLERANCE_LOG10}" \
    "${MISSING_ARRAY_SPEC}" \
    "${MISSING_COUNT}" \
    "${CHECKPOINT_TEMP_ROOT}" \
    "${CHECKPOINT_WALLTIME}" \
    "${CHECKPOINT_ARRAY_JOB_ID}" \
    "${CHECKPOINT_GATE_JOB_ID}" \
    "${ASSEMBLY_JOB_ID}" \
    "${KR_JOB_ID}" \
    "${FINAL_GATE_JOB_ID}" \
    "${STRATIGRAPHY_JOB_ID}" <<'PY'
from datetime import datetime, timezone
import json
import sys

(
    output_path,
    run_id,
    orchestration_commit,
    physics_commit,
    method_config,
    method_config_sha256,
    replay_tolerance_log10,
    missing_indices,
    missing_count,
    checkpoint_temp_root,
    checkpoint_walltime,
    checkpoint_job_id,
    checkpoint_gate_job_id,
    assembly_job_id,
    kr_job_id,
    final_gate_job_id,
    stratigraphy_job_id,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "status": "submitted",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "orchestration_commit": orchestration_commit,
    "physics_commit": physics_commit,
    "method_config": method_config,
    "method_config_sha256": method_config_sha256,
    "replay_tolerance_log10": float(replay_tolerance_log10),
    "replay_tolerance_semantics": "maximum_allowed_numerical_difference",
    "completed_checkpoint_policy": "preserve_valid_existing_markers",
    "missing_checkpoint_count": int(missing_count),
    "missing_checkpoint_indices": [
        int(value) for value in missing_indices.split(",")
    ],
    "checkpoint_temp_root": checkpoint_temp_root,
    "checkpoint_walltime": checkpoint_walltime,
    "jobs": {
        "checkpoint_array": checkpoint_job_id,
        "checkpoint_gate": checkpoint_gate_job_id,
        "assembly_array": assembly_job_id,
        "dynamic_kr_array": kr_job_id,
        "final_qa_gate": final_gate_job_id,
        "geology_stratigraphy_package": stratigraphy_job_id,
    },
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(json.dumps(manifest, indent=2))
PY

echo "Submitted missing-only continuation."
