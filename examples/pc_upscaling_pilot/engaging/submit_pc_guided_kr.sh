#!/usr/bin/env bash
# Submit one Pc-guided representative dynamic-Kr job for any geology/case set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"

GEOLOGY_ID="${GEOLOGY_ID:-${KR_DYN_GEOLOGY_ID:-}}"
CASE_IDS="${CASE_IDS:-${KR_DYN_CASE_IDS:-}}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
PC_RUN_ID="${PC_RUN_ID:-${SOURCE_RUN_ID}}"
if [[ -z "${GEOLOGY_ID}" || -z "${CASE_IDS}" || -z "${SOURCE_RUN_ID}" ]]; then
    echo "GEOLOGY_ID, CASE_IDS, and SOURCE_RUN_ID are required." >&2
    exit 2
fi

case_token="cases"
case_count=0
IFS=',' read -r -a case_values <<< "${CASE_IDS}"
for raw_value in "${case_values[@]}"; do
    value="${raw_value//[[:space:]]/}"
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "Invalid integer in CASE_IDS: ${raw_value}" >&2
        exit 2
    fi
    printf -v suffix '_%02d' "${value}"
    case_token+="${suffix}"
    case_count=$((case_count + 1))
done

RUN_MODE="${RUN_MODE:-full}"
RUN_ID="${RUN_ID:-${GEOLOGY_ID}_${case_token}_pc_guided_kr_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/runs/${RUN_ID}}"
REPLAY_ROOT="${REPLAY_ROOT:-${SCRATCH_ROOT}/runs/${SOURCE_RUN_ID}/full87_replay}"
PC_ROOT="${PC_ROOT:-${SCRATCH_ROOT}/runs/${PC_RUN_ID}/pc_ip}"

REPLAY_SUMMARY="${KR_DYN_REPLAY_SUMMARY_CSV:-${REPLAY_ROOT}/tables/replay_summary_with_full87_context_${GEOLOGY_ID}_${case_token}.csv}"
PC_CURVES="${KR_DYN_PRECOMPUTED_PC_CURVE_CSV:-${PC_ROOT}/curves/pc_curve_points_${GEOLOGY_ID}_${case_token}_ip_full87.csv}"
PC_NATIVE_CURVES="${KR_DYN_PRECOMPUTED_PC_NATIVE_CURVE_CSV:-${PC_ROOT}/curves/pc_native_curve_points_${GEOLOGY_ID}_${case_token}_ip_full87.csv}"
PC_SUMMARY="${KR_DYN_PRECOMPUTED_PC_SUMMARY_CSV:-${PC_ROOT}/tables/pc_curve_summary_${GEOLOGY_ID}_${case_token}_ip_full87.csv}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"

for required in "${REPLAY_SUMMARY}" "${PC_CURVES}" "${PC_NATIVE_CURVES}" "${PC_SUMMARY}"; do
    if [[ ! -f "${required}" ]]; then
        if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
            echo "Input will be produced by dependency ${DEPENDENCY_JOB_ID}: ${required}"
        else
            echo "Required completed input not found: ${required}" >&2
            exit 2
        fi
    fi
done

mkdir -p "${RUN_ROOT}"/{logs,slurm,tmp,status,diagnostics}

export RUN_MODE RUN_ID RUN_ROOT PROJECT_ROOT SCRATCH_ROOT
export FULL87_REPLAY_OUTPUT_ROOT="${REPLAY_ROOT}"
export KR_DYN_REPLAY_ROOT="${REPLAY_ROOT}"
export KR_DYN_REPLAY_SUMMARY_CSV="${REPLAY_SUMMARY}"
export KR_DYN_PRECOMPUTED_PC_CURVE_CSV="${PC_CURVES}"
export KR_DYN_PRECOMPUTED_PC_NATIVE_CURVE_CSV="${PC_NATIVE_CURVES}"
export KR_DYN_PRECOMPUTED_PC_SUMMARY_CSV="${PC_SUMMARY}"
export KR_DYN_OUTPUT_ROOT="${KR_DYN_OUTPUT_ROOT:-${RUN_ROOT}/kr_dyn_pc_guided}"
export KR_DYN_GEOLOGY_ID="${GEOLOGY_ID}"
export KR_DYN_CASE_IDS="${CASE_IDS}"
export KR_DYN_SELECTION_MODE="${KR_DYN_SELECTION_MODE:-median_swi}"
export KR_DYN_PC_PRESTEP_MODE="${KR_DYN_PC_PRESTEP_MODE:-precomputed}"
export KR_DYN_USE_PARALLEL="${KR_DYN_USE_PARALLEL:-1}"
export KR_DYN_NUM_WORKERS="${KR_DYN_NUM_WORKERS:-6}"
export KR_DYN_3D_NUM_THREADS="${KR_DYN_3D_NUM_THREADS:-1}"
export KR_DYN_1D_AD_SOLVER="${KR_DYN_1D_AD_SOLVER:-robust}"
export KR_DYN_LINEAR_SOLVER="${KR_DYN_LINEAR_SOLVER:-amgcl_require}"
export KR_DYN_1D_LINEAR_SOLVER="${KR_DYN_1D_LINEAR_SOLVER:-amgcl_auto}"
export KR_DYN_TIMESTEP_MODE="${KR_DYN_TIMESTEP_MODE:-paper}"
export KR_DYN_COREY_STEP="${KR_DYN_COREY_STEP:-0.2}"
export KR_DYN_MAX_ROWS="${KR_DYN_MAX_ROWS:-}"
if [[ -n "${KR_DYN_MAX_ROWS}" ]]; then
    export KR_DYN_ALLOW_PARTIAL_REPLAY="${KR_DYN_ALLOW_PARTIAL_REPLAY:-1}"
else
    export KR_DYN_ALLOW_PARTIAL_REPLAY="${KR_DYN_ALLOW_PARTIAL_REPLAY:-0}"
fi

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mit_amf_advanced_cpu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-mit_normal}"
SBATCH_QOS="${SBATCH_QOS:-}"
KR_TIME="${KR_TIME:-04:00:00}"
KR_CPUS="${KR_CPUS:-6}"
KR_MEM="${KR_MEM:-32G}"

args=(
    --parsable
    --account="${SBATCH_ACCOUNT}"
    --partition="${SBATCH_PARTITION}"
    --job-name="predict_kr_pcguided"
    --time="${KR_TIME}"
    --cpus-per-task="${KR_CPUS}"
    --mem="${KR_MEM}"
    --output="${RUN_ROOT}/slurm/%x_%j.out"
    --error="${RUN_ROOT}/slurm/%x_%j.err"
    --export=ALL,ENGAGING_WORKFLOW_DIR="${SCRIPT_DIR}"
)
if [[ -n "${SBATCH_QOS}" ]]; then
    args+=(--qos="${SBATCH_QOS}")
fi
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
    args+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
fi

job_id="$(sbatch "${args[@]}" "${SCRIPT_DIR}/run_case01_stage.sh" kr)"
selected_count=$((6 * case_count))

echo "Submitted Pc-guided dynamic-Kr stage"
echo "  job:             ${job_id}"
echo "  run root:        ${RUN_ROOT}"
echo "  geology:         ${GEOLOGY_ID}"
echo "  Level-3 cases:   ${CASE_IDS}"
echo "  replay source:   ${SOURCE_RUN_ID}"
echo "  Pc source:       ${PC_RUN_ID}"
echo "  selected curves: ${selected_count} (one median-Swi realization per window/case)"
echo "  resources:       ${KR_TIME}, ${KR_CPUS} CPUs, ${KR_MEM}"
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
    echo "  after job:        ${DEPENDENCY_JOB_ID}"
fi
