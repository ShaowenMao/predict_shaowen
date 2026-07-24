#!/usr/bin/env bash
# Assemble field-case inputs and submit dynamic-Kr production jobs.

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "qualification60" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 qualification60|full" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"

if [[ "${MODE}" == "qualification60" ]]; then
    RUN_ID="${RUN_ID:-production_qualification60_20260723_v1}"
    CASE_FILTER="${CASE_FILTER:-${SCRIPT_DIR}/production_qualification_60_cases.csv}"
    KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-12}"
else
    RUN_ID="${RUN_ID:-production_all1620_20260723_v1}"
    CASE_FILTER=""
    KR_MAX_CONCURRENT="${KR_MAX_CONCURRENT:-24}"
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-${RUN_ROOT}/checkpoint_pc}"
CASE_WORK_ROOT="${CASE_WORK_ROOT:-${RUN_ROOT}/case_work_manifest}"
CASE_INPUT_ROOT="${CASE_INPUT_ROOT:-${RUN_ROOT}/case_inputs}"
CASE_RESULT_ROOT="${CASE_RESULT_ROOT:-${RUN_ROOT}/case_results}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/production_logs/${RUN_ID}}"
SAMPLING_CSV="${FREEZE_ROOT}/inputs/sampling/texas_field_slice_window_values.csv"
ASSIGNMENT_CSV="${FREEZE_ROOT}/manifests/assignment_to_task.csv"
CHECKPOINT_JOB_ID="${CHECKPOINT_JOB_ID:-}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-1.0e-3}"

if [[ -z "${CHECKPOINT_JOB_ID}" && -f "${RUN_ROOT}/checkpoint_array_job_id.txt" ]]; then
    CHECKPOINT_JOB_ID="$(<"${RUN_ROOT}/checkpoint_array_job_id.txt")"
fi
if [[ -z "${CHECKPOINT_JOB_ID}" ]]; then
    echo "CHECKPOINT_JOB_ID is required." >&2
    exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}/assembly" "${LOG_ROOT}/kr"
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

if [[ ! -f "${CASE_WORK_ROOT}/case_work.csv" ]]; then
    build_args=(
        --sampling-csv "${SAMPLING_CSV}"
        --assignment-to-task-csv "${ASSIGNMENT_CSV}"
        --output-root "${CASE_WORK_ROOT}"
    )
    if [[ -n "${CASE_FILTER}" ]]; then
        build_args+=(--case-filter-csv "${CASE_FILTER}")
    fi
    python3 "${SCRIPT_DIR}/build_case_work_manifest.py" "${build_args[@]}"
fi

read -r GEOLOGY_COUNT CASE_COUNT < <(
    python3 - "${CASE_WORK_ROOT}/case_work_metadata.json" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(data["geology_count"], data["case_count"])
PY
)

checkpoint_state="$(sacct -X -j "${CHECKPOINT_JOB_ID}" -n -P -o State | head -n 1 | cut -d+ -f1)"
dependency=()
case "${checkpoint_state}" in
    COMPLETED)
        echo "Checkpoint job ${CHECKPOINT_JOB_ID} already completed."
        ;;
    PENDING|RUNNING|CONFIGURING|COMPLETING)
        dependency=(--dependency="afterok:${CHECKPOINT_JOB_ID}")
        ;;
    *)
        echo "Checkpoint job ${CHECKPOINT_JOB_ID} is not usable: ${checkpoint_state}" >&2
        exit 2
        ;;
esac

assembly_submission="$(
    sbatch \
        --parsable \
        --account=mit_amf_advanced_cpu \
        --qos="${SLURM_QOS}" \
        --partition=mit_normal \
        --job-name="asm_${RUN_ID}" \
        --time="${ASSEMBLY_WALLTIME:-02:00:00}" \
        --cpus-per-task=1 \
        --mem="${ASSEMBLY_MEMORY:-16G}" \
        --array="1-${GEOLOGY_COUNT}%${ASSEMBLY_MAX_CONCURRENT:-12}" \
        --output="${LOG_ROOT}/assembly/%x_%A_%a.out" \
        --error="${LOG_ROOT}/assembly/%x_%A_%a.err" \
        "${dependency[@]}" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",CASE_WORK_ROOT="${CASE_WORK_ROOT}",CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT}",CASE_INPUT_ROOT="${CASE_INPUT_ROOT}" \
        "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_assemble_geology_cases.sh"
)"
ASSEMBLY_JOB_ID="${assembly_submission%%;*}"

kr_submission="$(
    sbatch \
        --parsable \
        --account=mit_amf_advanced_cpu \
        --qos="${SLURM_QOS}" \
        --partition=mit_normal \
        --job-name="kr_${RUN_ID}" \
        --time="${KR_WALLTIME:-08:00:00}" \
        --cpus-per-task=6 \
        --mem="${KR_MEMORY:-48G}" \
        --array="1-${CASE_COUNT}%${KR_MAX_CONCURRENT}" \
        --dependency="afterok:${ASSEMBLY_JOB_ID}" \
        --output="${LOG_ROOT}/kr/%x_%A_%a.out" \
        --error="${LOG_ROOT}/kr/%x_%A_%a.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",FREEZE_ROOT="${FREEZE_ROOT}",CASE_WORK_ROOT="${CASE_WORK_ROOT}",CASE_INPUT_ROOT="${CASE_INPUT_ROOT}",CASE_RESULT_ROOT="${CASE_RESULT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10}" \
        "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_case_dynamic_kr.sh"
)"
KR_JOB_ID="${kr_submission%%;*}"

cat > "${RUN_ROOT}/case_submission.env" <<EOF
run_id=${RUN_ID}
mode=${MODE}
submitted_at=$(date --iso-8601=seconds)
checkpoint_job_id=${CHECKPOINT_JOB_ID}
assembly_job_id=${ASSEMBLY_JOB_ID}
kr_job_id=${KR_JOB_ID}
geology_count=${GEOLOGY_COUNT}
case_count=${CASE_COUNT}
kr_max_concurrent=${KR_MAX_CONCURRENT}
replay_tolerance_log10=${REPLAY_TOLERANCE_LOG10}
slurm_qos=${SLURM_QOS}
EOF
echo "${ASSEMBLY_JOB_ID}" > "${RUN_ROOT}/assembly_array_job_id.txt"
echo "${KR_JOB_ID}" > "${RUN_ROOT}/kr_array_job_id.txt"
echo "Submitted assembly job ${ASSEMBLY_JOB_ID} (${GEOLOGY_COUNT} geologies)."
echo "Submitted dynamic-Kr job ${KR_JOB_ID} (${CASE_COUNT} cases)."
