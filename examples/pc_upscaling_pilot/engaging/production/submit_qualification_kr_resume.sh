#!/usr/bin/env bash
# Resume only missing dynamic-Kr stages in a completed qualification replay/Pc batch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v3}"
BATCH_ID="${BATCH_ID:-qualification_ccu_20260722_v3b_fullgrid}"
BATCH_ROOT="${BATCH_ROOT:-${SCRATCH_ROOT}/runs/${BATCH_ID}}"
CASES_FILE="${QUALIFICATION_CASES_FILE:-${SCRIPT_DIR}/qualification_cases.csv}"
FROZEN_REPO="${FREEZE_ROOT}/code/source"
FROZEN_WORKFLOW_DIR="${FROZEN_REPO}/examples/pc_upscaling_pilot/engaging"
STAGE_SCRIPT="${QUALIFICATION_STAGE_SCRIPT:-${FROZEN_WORKFLOW_DIR}/run_case01_stage.sh}"
SAMPLING_ROOT="${FREEZE_ROOT}/inputs/sampling"
PREDICT_ROOT="${FREEZE_ROOT}/inputs/predict"
SAMPLING_MAT="${SAMPLING_ROOT}/texas_field_sampling_compact.mat"
MRST_ROOT="${MRST_ROOT:-${PROJECT_ROOT}/software/mrst-current}"
UPSCALING_ZIP="${UPSCALING_ZIP:-${PROJECT_ROOT}/software/upscaling.zip}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mit_amf_advanced_cpu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-mit_normal}"
SBATCH_QOS="${SBATCH_QOS:-}"
SBATCH_EXCLUDE_NODES="${SBATCH_EXCLUDE_NODES:-}"
KR_TIME="${KR_TIME:-08:00:00}"
KR_CPUS="${KR_CPUS:-6}"
KR_MEM="${KR_MEM:-48G}"
DRY_RUN="${DRY_RUN:-0}"

COMMIT_FILE="${FREEZE_ROOT}/code/COMMIT.txt"
METADATA_FILE="${FREEZE_ROOT}/freeze_metadata.json"
RESUBMISSION_MANIFEST="${BATCH_ROOT}/kr_resubmission_manifest.csv"

require_file() {
    [[ -f "$1" ]] || { echo "Missing required file: $1" >&2; exit 2; }
}

require_file "${CASES_FILE}"
require_file "${STAGE_SCRIPT}"
require_file "${SAMPLING_MAT}"
require_file "${UPSCALING_ZIP}"
require_file "${MRST_ROOT}/startup.m"
require_file "${COMMIT_FILE}"
require_file "${METADATA_FILE}"

FROZEN_COMMIT="$(tr -d '[:space:]' < "${COMMIT_FILE}")"
grep -q "${FROZEN_COMMIT}" "${METADATA_FILE}" || {
    echo "Freeze metadata does not contain code commit ${FROZEN_COMMIT}." >&2
    exit 2
}
find -H "${MRST_ROOT}" -type f -name 'amgcl_matlab*.mexa64' -print -quit | grep -q . || {
    echo "AMGCL MEX was not found beneath ${MRST_ROOT}." >&2
    exit 2
}

mkdir -p "${BATCH_ROOT}"
if [[ ! -e "${RESUBMISSION_MANIFEST}" ]]; then
    echo 'submitted_at,scenario_index,geology_id,case_id,run_root,kr_job_id,code_commit,freeze_root,stage_script_sha256' \
        > "${RESUBMISSION_MANIFEST}"
fi
STAGE_SCRIPT_SHA256="$(sha256sum "${STAGE_SCRIPT}" | awk '{print $1}')"

submit_kr() {
    local scenario_index="$1"
    local geology_id="$2"
    local case_id="$3"
    local case_two_digit run_root prefix job_id submission
    printf -v case_two_digit '%02d' "${case_id}"
    run_root="${BATCH_ROOT}/cases/${geology_id}/case${case_two_digit}"
    prefix="qccu_s${scenario_index}c012_c${case_two_digit}_k"

    require_file "${run_root}/status/replay.done"
    require_file "${run_root}/status/pc.done"
    require_file "${BATCH_ROOT}/inputs/${geology_id}_case${case_two_digit}_slice_window_values.csv"
    if [[ -e "${run_root}/status/kr.done" ]]; then
        echo "Skipping completed Kr stage: ${geology_id} case ${case_two_digit}"
        return
    fi

    mkdir -p "${run_root}"/{logs,slurm,tmp,status,diagnostics}
    local export_vars
    export_vars="ALL"
    export_vars+=",ENGAGING_WORKFLOW_DIR=${FROZEN_WORKFLOW_DIR}"
    export_vars+=",PROJECT_ROOT=${PROJECT_ROOT},SCRATCH_ROOT=${SCRATCH_ROOT}"
    export_vars+=",RUN_MODE=full,RUN_ID=${BATCH_ID}_${geology_id}_case${case_two_digit},RUN_ROOT=${run_root}"
    export_vars+=",FULL87_REPLAY_OUTPUT_ROOT=${run_root}/full87_replay"
    export_vars+=",FULL87_REPLAY_FIELD_SAMPLING_CSV=${BATCH_ROOT}/inputs/${geology_id}_case${case_two_digit}_slice_window_values.csv"
    export_vars+=",KR_DYN_PERMEABILITY_INPUT=${SAMPLING_MAT}"
    export_vars+=",FULL87_REPLAY_DATA_ROOT=${PREDICT_ROOT},PREDICT_REPLAY_CODE_ROOT=${FROZEN_REPO}"
    export_vars+=",FULL87_REPLAY_GEOLOGY_ID=${geology_id},FULL87_REPLAY_CASE_IDS=${case_id}"
    export_vars+=",PC_IP_GEOLOGY_ID=${geology_id},PC_IP_CASE_IDS=${case_id}"
    export_vars+=",PC_IP_REPLAY_ROOT=${run_root}/full87_replay,PC_IP_OUTPUT_ROOT=${run_root}/pc_ip"
    export_vars+=",KR_DYN_GEOLOGY_ID=${geology_id},KR_DYN_CASE_IDS=${case_id}"
    export_vars+=",KR_DYN_REPLAY_ROOT=${run_root}/full87_replay,KR_DYN_OUTPUT_ROOT=${run_root}/kr_dyn_swi_medoid"
    export_vars+=",KR_DYN_SELECTION_MODE=swi_medoid,KR_DYN_EXPORT_RESERVOIR_READY=1"
    export_vars+=",KR_DYN_PC_PRESTEP_MODE=precomputed,KR_DYN_USE_PARALLEL=1,KR_DYN_NUM_WORKERS=${KR_CPUS}"
    export_vars+=",KR_DYN_3D_NUM_THREADS=1,KR_DYN_1D_AD_SOLVER=robust"
    export_vars+=",KR_DYN_LINEAR_SOLVER=amgcl_require,KR_DYN_1D_LINEAR_SOLVER=amgcl_auto"
    export_vars+=",KR_DYN_TIMESTEP_MODE=paper,KR_DYN_COREY_STEP=0.2"

    local args=(
        --parsable
        --account="${SBATCH_ACCOUNT}"
        --partition="${SBATCH_PARTITION}"
        --job-name="${prefix}"
        --time="${KR_TIME}"
        --cpus-per-task="${KR_CPUS}"
        --mem="${KR_MEM}"
        --output="${run_root}/slurm/%x_%j.out"
        --error="${run_root}/slurm/%x_%j.err"
        --export="${export_vars}"
    )
    if [[ -n "${SBATCH_QOS}" ]]; then
        args+=(--qos="${SBATCH_QOS}")
    fi
    if [[ -n "${SBATCH_EXCLUDE_NODES}" ]]; then
        args+=(--exclude="${SBATCH_EXCLUDE_NODES}")
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "DRY RUN: ${geology_id} case ${case_two_digit}"
        return
    fi
    submission="$(sbatch "${args[@]}" "${STAGE_SCRIPT}" kr)"
    job_id="${submission%%;*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || {
        echo "Invalid Slurm job ID returned for ${geology_id} case ${case_two_digit}: ${submission}" >&2
        exit 2
    }
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(date --iso-8601=seconds)" "${scenario_index}" "${geology_id}" \
        "${case_id}" "${run_root}" "${job_id}" "${FROZEN_COMMIT}" \
        "${FREEZE_ROOT}" "${STAGE_SCRIPT_SHA256}" >> "${RESUBMISSION_MANIFEST}"
    echo "Submitted Kr: ${geology_id} case ${case_two_digit}, job ${job_id}"
}

while IFS=, read -r scenario_index geology_id _scenario_label case_id _case_category; do
    [[ "${scenario_index}" == "scenario_index" ]] && continue
    submit_kr "${scenario_index}" "${geology_id}" "${case_id}"
done < "${CASES_FILE}"

echo "Kr resubmission manifest: ${RESUBMISSION_MANIFEST}"
