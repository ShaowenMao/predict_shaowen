#!/usr/bin/env bash
# Submit a marker-aware dynamic-Kr recovery after the original array finishes.

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
ORIGINAL_KR_JOB_ID="${ORIGINAL_KR_JOB_ID:-}"
KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-24}"
KR_WALLTIME="${KR_WALLTIME:-12:00:00}"
KR_MEMORY="${KR_MEMORY:-48G}"
CASE_COUNT=1620
CASES_PER_ARRAY_TASK=10
KR_ARRAY_TASK_COUNT=162

CASE_WORK_ROOT="${RUN_ROOT}/case_work_manifest"
CASE_INPUT_ROOT="${RUN_ROOT}/case_inputs"
CASE_RESULT_ROOT="${RUN_ROOT}/case_results"
LOG_ROOT="${SCRATCH_ROOT}/production_logs/${RUN_ID}"

if [[ -z "${ORIGINAL_KR_JOB_ID}" && -f "${RUN_ROOT}/kr_array_job_id.txt" ]]; then
    ORIGINAL_KR_JOB_ID="$(<"${RUN_ROOT}/kr_array_job_id.txt")"
fi
[[ -n "${ORIGINAL_KR_JOB_ID}" ]] || {
    echo "ORIGINAL_KR_JOB_ID is required." >&2
    exit 2
}
[[ -f "${CASE_WORK_ROOT}/case_work.csv" ]] || {
    echo "Missing case work manifest: ${CASE_WORK_ROOT}/case_work.csv" >&2
    exit 2
}
if [[ "${ACTION}" == "submit" && -z "${ORCHESTRATION_COMMIT}" ]]; then
    echo "ORCHESTRATION_COMMIT is required for submission provenance." >&2
    exit 2
fi
if [[ "${ACTION}" == "submit" && -f "${RUN_ROOT}/kr_recovery_job_id.txt" ]]; then
    previous_recovery_job_id="$(<"${RUN_ROOT}/kr_recovery_job_id.txt")"
    if [[ -n "${previous_recovery_job_id}" ]] && \
            [[ -n "$(squeue -h -j "${previous_recovery_job_id}" 2>/dev/null)" ]]; then
        echo "Recovery job ${previous_recovery_job_id} is already active." >&2
        exit 2
    fi
fi

completed_cases="$(
    find "${CASE_RESULT_ROOT}" -type f -name case.done.json 2>/dev/null | wc -l
)"
cat <<EOF
Dynamic-Kr recovery plan
  run_id: ${RUN_ID}
  completed case markers now: ${completed_cases}/${CASE_COUNT}
  original Kr array: ${ORIGINAL_KR_JOB_ID}
  recovery dependency: afterany:${ORIGINAL_KR_JOB_ID}
  recovery strategy: all ${KR_ARRAY_TASK_COUNT} geology chunks, marker-aware skips
  recovery concurrency: ${KR_MAX_CONCURRENT}
  recovery walltime: ${KR_WALLTIME}
  orchestration commit: ${ORCHESTRATION_COMMIT:-not-required-for-plan}
  physics commit: ${PHYSICS_COMMIT}
EOF

if [[ "${ACTION}" == "plan" ]]; then
    exit 0
fi

mkdir -p "${LOG_ROOT}/kr_recovery" "${LOG_ROOT}/final_gate_recovery"

recovery_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="krr_${RUN_ID}" \
        --time="${KR_WALLTIME}" \
        --cpus-per-task=6 \
        --mem="${KR_MEMORY}" \
        --array="1-${KR_ARRAY_TASK_COUNT}%${KR_MAX_CONCURRENT}" \
        --dependency="afterany:${ORIGINAL_KR_JOB_ID}" \
        --output="${LOG_ROOT}/kr_recovery/%x_%A_%a.out" \
        --error="${LOG_ROOT}/kr_recovery/%x_%A_%a.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",FREEZE_ROOT="${FREEZE_ROOT}",CASE_WORK_ROOT="${CASE_WORK_ROOT}",CASE_INPUT_ROOT="${CASE_INPUT_ROOT}",CASE_RESULT_ROOT="${CASE_RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}",CASE_COUNT="${CASE_COUNT}",CASES_PER_ARRAY_TASK="${CASES_PER_ARRAY_TASK}" \
        "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_case_dynamic_kr_chunk.sh"
)"
RECOVERY_JOB_ID="${recovery_submission%%;*}"

final_gate_submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="qar_${RUN_ID}" \
        --time="${FINAL_GATE_WALLTIME:-08:00:00}" \
        --cpus-per-task=1 \
        --mem="${FINAL_GATE_MEMORY:-8G}" \
        --dependency="afterany:${RECOVERY_JOB_ID}" \
        --output="${LOG_ROOT}/final_gate_recovery/%x_%j.out" \
        --error="${LOG_ROOT}/final_gate_recovery/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",RUN_ROOT="${RUN_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}",MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${REPLAY_TOLERANCE_LOG10}" \
        "${SCRIPT_DIR}/run_case_completion_gate.sh"
)"
FINAL_GATE_JOB_ID="${final_gate_submission%%;*}"

echo "${RECOVERY_JOB_ID}" > "${RUN_ROOT}/kr_recovery_job_id.txt"
echo "${FINAL_GATE_JOB_ID}" > "${RUN_ROOT}/final_qa_job_id.txt"

python3 - \
    "${RUN_ROOT}/dynamic_kr_recovery_manifest.json" \
    "${RUN_ID}" \
    "${ORCHESTRATION_COMMIT}" \
    "${ORIGINAL_KR_JOB_ID}" \
    "${RECOVERY_JOB_ID}" \
    "${FINAL_GATE_JOB_ID}" \
    "${completed_cases}" \
    "${KR_MAX_CONCURRENT}" \
    "${PHYSICS_COMMIT}" \
    "${METHOD_CONFIG_SHA256}" <<'PY'
from datetime import datetime, timezone
import json
import sys

(
    output_path,
    run_id,
    orchestration_commit,
    original_job_id,
    recovery_job_id,
    final_gate_job_id,
    completed_cases,
    max_concurrent,
    physics_commit,
    method_config_sha256,
) = sys.argv[1:]

manifest = {
    "schema_version": 1,
    "status": "submitted",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "orchestration_commit": orchestration_commit,
    "physics_commit": physics_commit,
    "method_config_sha256": method_config_sha256,
    "strategy": "all_geology_chunks_marker_aware",
    "case_count": 1620,
    "completed_case_markers_at_submission": int(completed_cases),
    "max_concurrent": int(max_concurrent),
    "jobs": {
        "original_dynamic_kr_array": original_job_id,
        "recovery_dynamic_kr_array": recovery_job_id,
        "final_qa_gate": final_gate_job_id,
    },
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
PY

echo "Submitted marker-aware dynamic-Kr recovery ${RECOVERY_JOB_ID}."
echo "Submitted replacement final QA gate ${FINAL_GATE_JOB_ID}."
