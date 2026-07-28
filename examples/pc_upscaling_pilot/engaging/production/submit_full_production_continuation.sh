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
CHECKPOINT_MEMORY="${CHECKPOINT_MEMORY:-16G}"
CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT:-${SCRATCH_ROOT}/tmp}"
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

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

mapfile -t missing_indices < <(
    python3 - "${GROUPS_CSV}" "${CHECKPOINT_OUTPUT_ROOT}" <<'PY'
import csv
import json
import sys
from pathlib import Path

groups_csv = Path(sys.argv[1])
output_root = Path(sys.argv[2])
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
    if data.get("status") != "complete" or data.get("group_id") != row["group_id"]:
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
    "${LOG_ROOT}/final_gate_continuation" \
    "${CHECKPOINT_TEMP_ROOT}"

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
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",FREEZE_ROOT="${FREEZE_ROOT}",CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT}",COMPACT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",CHECKPOINT_TEMP_ROOT="${CHECKPOINT_TEMP_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",GROUP_COUNT="${GROUP_COUNT}",GROUPS_PER_ARRAY_TASK=1 \
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
