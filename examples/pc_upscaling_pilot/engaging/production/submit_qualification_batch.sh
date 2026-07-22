#!/usr/bin/env bash
# Submit the updated-data qualification replay -> Pc -> dynamic-Kr chains.

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 smoke|full" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v1}"
FROZEN_REPO="${FREEZE_ROOT}/code/source"
FROZEN_WORKFLOW_DIR="${FROZEN_REPO}/examples/pc_upscaling_pilot/engaging"
STAGE_SCRIPT="${FROZEN_WORKFLOW_DIR}/run_case01_stage.sh"
CASES_FILE="${QUALIFICATION_CASES_FILE:-${SCRIPT_DIR}/qualification_cases.csv}"

EXPECTED_COMMIT="344dd97ae9a9ac244e82c83a443a208dbd798788"
EXPECTED_METHOD_HASH="efa43fca01ccb2929caffab4be79cabd4d148d88539e4e78c6e109145c31e33b"
EXPECTED_FIELD_CONFIG_HASH="aee718ad0c43a2d16087980a848a8abacdc5893a1237088c51378666a027b7e6"

SAMPLING_ROOT="${FREEZE_ROOT}/inputs/sampling"
PREDICT_ROOT="${FREEZE_ROOT}/inputs/predict"
SAMPLING_CSV="${SAMPLING_ROOT}/texas_field_slice_window_values.csv"
SAMPLING_MAT="${SAMPLING_ROOT}/texas_field_sampling_compact.mat"
METHOD_CONFIG="${FREEZE_ROOT}/config/production_method_config.toml"
FIELD_CONFIG="${FREEZE_ROOT}/config/texas_field_collapsed_cell_union_config.toml"
MRST_ROOT="${MRST_ROOT:-${PROJECT_ROOT}/software/mrst-current}"
UPSCALING_ZIP="${UPSCALING_ZIP:-${PROJECT_ROOT}/software/upscaling.zip}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mit_amf_advanced_cpu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-mit_normal}"
SBATCH_QOS="${SBATCH_QOS:-}"
QUALIFICATION_GATE_JOB_ID="${QUALIFICATION_GATE_JOB_ID:-}"
RESUME="${RESUME:-0}"

if [[ "${MODE}" == "smoke" ]]; then
    BATCH_ID="${BATCH_ID:-qualification_ccu_20260722_v1_smoke_gate}"
    REPLAY_TIME="${REPLAY_TIME:-00:30:00}"
    REPLAY_CPUS="${REPLAY_CPUS:-1}"
    REPLAY_MEM="${REPLAY_MEM:-8G}"
    PC_TIME="${PC_TIME:-00:45:00}"
    PC_CPUS="${PC_CPUS:-1}"
    PC_MEM="${PC_MEM:-12G}"
    KR_TIME="${KR_TIME:-01:30:00}"
    KR_CPUS="${KR_CPUS:-2}"
    KR_MEM="${KR_MEM:-16G}"
else
    BATCH_ID="${BATCH_ID:-qualification_ccu_20260722_v1_full}"
    REPLAY_TIME="${REPLAY_TIME:-03:00:00}"
    REPLAY_CPUS="${REPLAY_CPUS:-1}"
    REPLAY_MEM="${REPLAY_MEM:-16G}"
    PC_TIME="${PC_TIME:-03:00:00}"
    PC_CPUS="${PC_CPUS:-1}"
    PC_MEM="${PC_MEM:-24G}"
    KR_TIME="${KR_TIME:-08:00:00}"
    KR_CPUS="${KR_CPUS:-6}"
    KR_MEM="${KR_MEM:-48G}"
fi

BATCH_ROOT="${BATCH_ROOT:-${SCRATCH_ROOT}/runs/${BATCH_ID}}"
SUBMISSION_MANIFEST="${BATCH_ROOT}/submission_manifest.csv"
QUALIFICATION_INPUT_ROOT="${BATCH_ROOT}/inputs"

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

preflight() {
    require_file "${FREEZE_ROOT}/freeze_metadata.json"
    require_file "${FREEZE_ROOT}/code/predict_shaowen_344dd97.bundle"
    require_file "${METHOD_CONFIG}"
    require_file "${FIELD_CONFIG}"
    require_file "${SAMPLING_CSV}"
    require_file "${SAMPLING_MAT}"
    require_file "${PREDICT_ROOT}/geology_case_definitions.csv"
    require_file "${PREDICT_ROOT}/thickness_scenario_definitions.csv"
    require_file "${STAGE_SCRIPT}"
    require_file "${CASES_FILE}"
    require_file "${UPSCALING_ZIP}"
    require_file "${MRST_ROOT}/startup.m"

    grep -q "${EXPECTED_COMMIT}" "${FREEZE_ROOT}/freeze_metadata.json" || {
        echo "Frozen code commit does not match ${EXPECTED_COMMIT}." >&2
        exit 2
    }
    echo "${EXPECTED_METHOD_HASH}  ${METHOD_CONFIG}" | sha256sum --check --status || {
        echo "Method configuration hash mismatch." >&2
        exit 2
    }
    echo "${EXPECTED_FIELD_CONFIG_HASH}  ${FIELD_CONFIG}" | sha256sum --check --status || {
        echo "Field configuration hash mismatch." >&2
        exit 2
    }
    find -H "${MRST_ROOT}" -type f -name 'amgcl_matlab*.mexa64' -print -quit | grep -q . || {
        echo "AMGCL MEX was not found beneath ${MRST_ROOT}." >&2
        exit 2
    }
}

submit_stage() {
    local stage="$1"
    local job_name="$2"
    local time_limit="$3"
    local cpus="$4"
    local memory="$5"
    local dependency="${6:-}"
    local args=(
        --parsable
        --account="${SBATCH_ACCOUNT}"
        --partition="${SBATCH_PARTITION}"
        --job-name="${job_name}"
        --time="${time_limit}"
        --cpus-per-task="${cpus}"
        --mem="${memory}"
        --output="${RUN_ROOT}/slurm/%x_%j.out"
        --error="${RUN_ROOT}/slurm/%x_%j.err"
        --export=ALL,ENGAGING_WORKFLOW_DIR="${FROZEN_WORKFLOW_DIR}"
    )
    if [[ -n "${dependency}" ]]; then
        args+=(--dependency="afterok:${dependency}")
    fi
    if [[ -n "${SBATCH_QOS}" ]]; then
        args+=(--qos="${SBATCH_QOS}")
    fi
    local submission
    submission="$(sbatch "${args[@]}" "${STAGE_SCRIPT}" "${stage}")"
    printf '%s\n' "${submission%%;*}"
}

configure_case() {
    local geology_id="$1"
    local case_id="$2"
    printf -v case_two_digit '%02d' "${case_id}"
    RUN_ID="${BATCH_ID}_${geology_id}_case${case_two_digit}"
    RUN_ROOT="${BATCH_ROOT}/cases/${geology_id}/case${case_two_digit}"

    if [[ -e "${RUN_ROOT}" && "${RESUME}" != "1" ]]; then
        echo "Run root exists; set RESUME=1 to continue: ${RUN_ROOT}" >&2
        exit 2
    fi
    mkdir -p "${RUN_ROOT}"/{logs,slurm,tmp,status,diagnostics}

    export PROJECT_ROOT SCRATCH_ROOT RUN_ID RUN_ROOT MRST_ROOT UPSCALING_ZIP
    export RUN_MODE="${MODE}"
    export FULL87_REPLAY_OUTPUT_ROOT="${RUN_ROOT}/full87_replay"
    export FULL87_REPLAY_FIELD_SAMPLING_CSV="${QUALIFICATION_INPUT_ROOT}/${geology_id}_case${case_two_digit}_slice_window_values.csv"
    require_file "${FULL87_REPLAY_FIELD_SAMPLING_CSV}"
    export KR_DYN_PERMEABILITY_INPUT="${SAMPLING_MAT}"
    export FULL87_REPLAY_DATA_ROOT="${PREDICT_ROOT}"
    export PREDICT_REPLAY_CODE_ROOT="${FROZEN_REPO}"
    export FULL87_REPLAY_GEOLOGY_ID="${geology_id}"
    export FULL87_REPLAY_CASE_IDS="${case_id}"
    export FULL87_REPLAY_VERIFY_TOLERANCE_LOG10="1.0e-8"
    export PC_IP_GEOLOGY_ID="${geology_id}"
    export PC_IP_CASE_IDS="${case_id}"
    export PC_IP_REPLAY_ROOT="${FULL87_REPLAY_OUTPUT_ROOT}"
    export PC_IP_OUTPUT_ROOT="${RUN_ROOT}/pc_ip"
    export PC_IP_ENABLE_MEDOID_DIAGNOSTICS="0"
    if [[ "${MODE}" == "smoke" ]]; then
        # Process every row in the intentionally truncated replay table while
        # retaining the normal pc_ip path expected by the downstream Kr stage.
        export PC_IP_MAX_ROWS="Inf"
    fi
    export KR_DYN_GEOLOGY_ID="${geology_id}"
    export KR_DYN_CASE_IDS="${case_id}"
    export KR_DYN_REPLAY_ROOT="${FULL87_REPLAY_OUTPUT_ROOT}"
    export KR_DYN_OUTPUT_ROOT="${RUN_ROOT}/kr_dyn_swi_medoid"
    export KR_DYN_SELECTION_MODE="swi_medoid"
    export KR_DYN_EXPORT_RESERVOIR_READY="1"
    export KR_DYN_PC_PRESTEP_MODE="precomputed"
    export KR_DYN_USE_PARALLEL="1"
    export KR_DYN_NUM_WORKERS="${KR_CPUS}"
    export KR_DYN_3D_NUM_THREADS="1"
    export KR_DYN_1D_AD_SOLVER="robust"
    export KR_DYN_LINEAR_SOLVER="amgcl_require"
    export KR_DYN_1D_LINEAR_SOLVER="amgcl_auto"
    export KR_DYN_TIMESTEP_MODE="paper"
    export KR_DYN_COREY_STEP="0.2"
}

submit_case_chain() {
    local scenario_index="$1"
    local geology_id="$2"
    local scenario_name="$3"
    local case_id="$4"
    local case_category="$5"
    local initial_dependency="${6:-}"

    configure_case "${geology_id}" "${case_id}"
    printf -v case_two_digit '%02d' "${case_id}"
    local prefix="qccu_s${scenario_index}c012_c${case_two_digit}"
    local replay_job pc_job kr_job
    replay_job="$(submit_stage replay "${prefix}_r" "${REPLAY_TIME}" "${REPLAY_CPUS}" "${REPLAY_MEM}" "${initial_dependency}")"
    pc_job="$(submit_stage pc "${prefix}_p" "${PC_TIME}" "${PC_CPUS}" "${PC_MEM}" "${replay_job}")"
    kr_job="$(submit_stage kr "${prefix}_k" "${KR_TIME}" "${KR_CPUS}" "${KR_MEM}" "${pc_job}")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${scenario_index}" "${geology_id}" "${scenario_name}" "${case_id}" \
        "${case_category}" "${RUN_ID}" "${RUN_ROOT}" "${replay_job}" \
        "${pc_job}" "${kr_job}" "${initial_dependency}" "${EXPECTED_COMMIT}" \
        >> "${SUBMISSION_MANIFEST}"
    echo "Submitted ${geology_id} case ${case_two_digit}: replay=${replay_job}, Pc=${pc_job}, Kr=${kr_job}"
}

preflight
mkdir -p "${BATCH_ROOT}"
if [[ -e "${SUBMISSION_MANIFEST}" && "${RESUME}" != "1" ]]; then
    echo "Submission manifest exists; set RESUME=1 to append a continuation." >&2
    exit 2
fi
if [[ ! -e "${SUBMISSION_MANIFEST}" ]]; then
    echo 'scenario_index,geology_id,scenario_name,case_id,case_category,run_id,run_root,replay_job_id,pc_job_id,kr_job_id,gate_job_id,code_commit' > "${SUBMISSION_MANIFEST}"
fi

cp "${CASES_FILE}" "${BATCH_ROOT}/qualification_cases.csv"
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64
python3 "${SCRIPT_DIR}/prepare_qualification_inputs.py" \
    --sampling-csv "${SAMPLING_CSV}" \
    --cases-csv "${CASES_FILE}" \
    --output-root "${QUALIFICATION_INPUT_ROOT}"
cat > "${BATCH_ROOT}/batch_provenance.env" <<EOF
batch_id=${BATCH_ID}
mode=${MODE}
created_at=$(date --iso-8601=seconds)
freeze_root=${FREEZE_ROOT}
code_commit=${EXPECTED_COMMIT}
method_config_sha256=${EXPECTED_METHOD_HASH}
field_config_sha256=${EXPECTED_FIELD_CONFIG_HASH}
sampling_csv=${SAMPLING_CSV}
predict_root=${PREDICT_ROOT}
account=${SBATCH_ACCOUNT}
partition=${SBATCH_PARTITION}
EOF

if [[ "${MODE}" == "smoke" ]]; then
    submit_case_chain 1 s01_c012 "low sand uniform smoke" 1 "Independent cases" ""
    smoke_job="$(tail -n 1 "${SUBMISSION_MANIFEST}" | cut -d, -f10)"
    echo "${smoke_job}" > "${BATCH_ROOT}/smoke_gate_job_id.txt"
    echo "QUALIFICATION_GATE_JOB_ID=${smoke_job}"
else
    if [[ -z "${QUALIFICATION_GATE_JOB_ID}" ]]; then
        echo "QUALIFICATION_GATE_JOB_ID is required for the full qualification batch." >&2
        exit 2
    fi
    while IFS=, read -r scenario_index geology_id scenario_label case_id case_category; do
        [[ "${scenario_index}" == "scenario_index" ]] && continue
        submit_case_chain "${scenario_index}" "${geology_id}" \
            "${scenario_label}" "${case_id}" "${case_category}" \
            "${QUALIFICATION_GATE_JOB_ID}"
    done < "${CASES_FILE}"
fi

echo "Submission manifest: ${SUBMISSION_MANIFEST}"
echo "Batch root: ${BATCH_ROOT}"
