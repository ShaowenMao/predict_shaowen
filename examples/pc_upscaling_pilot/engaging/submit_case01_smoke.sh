#!/usr/bin/env bash
# Submit a chained replay -> Pc -> dynamic Kr workflow for case01.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_MODE="${RUN_MODE:-smoke}"
export RUN_ID="${RUN_ID:-case01_${RUN_MODE}_$(date +%Y%m%d_%H%M%S)}"
source "${SCRIPT_DIR}/env_case01.sh"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-mit_amf_advanced_cpu}"
SBATCH_PARTITION="${SBATCH_PARTITION:-mit_normal}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
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
    REPLAY_TIME="${REPLAY_TIME:-03:00:00}"
    REPLAY_CPUS="${REPLAY_CPUS:-1}"
    REPLAY_MEM="${REPLAY_MEM:-16G}"
    PC_TIME="${PC_TIME:-03:00:00}"
    PC_CPUS="${PC_CPUS:-1}"
    PC_MEM="${PC_MEM:-24G}"
    KR_TIME="${KR_TIME:-12:00:00}"
    KR_CPUS="${KR_CPUS:-12}"
    KR_MEM="${KR_MEM:-96G}"
fi

submit_stage() {
    local stage="$1"
    local name="$2"
    local time_limit="$3"
    local cpus="$4"
    local mem="$5"
    local dependency="${6:-}"

    local args=(
        --parsable
        --account="${SBATCH_ACCOUNT}"
        --partition="${SBATCH_PARTITION}"
        --job-name="${name}"
        --time="${time_limit}"
        --cpus-per-task="${cpus}"
        --mem="${mem}"
        --output="${RUN_ROOT}/slurm/%x_%j.out"
        --error="${RUN_ROOT}/slurm/%x_%j.err"
        --export=ALL,RUN_ID="${RUN_ID}",RUN_MODE="${RUN_MODE}",ENGAGING_WORKFLOW_DIR="${SCRIPT_DIR}"
    )
    if [[ -n "${dependency}" ]]; then
        args+=(--dependency="afterok:${dependency}")
    fi

    sbatch "${args[@]}" "${SCRIPT_DIR}/run_case01_stage.sh" "${stage}"
}

replay_job="$(submit_stage replay predict_c01_replay "${REPLAY_TIME}" "${REPLAY_CPUS}" "${REPLAY_MEM}")"
pc_job="$(submit_stage pc predict_c01_pc_ip "${PC_TIME}" "${PC_CPUS}" "${PC_MEM}" "${replay_job}")"
kr_job="$(submit_stage kr predict_c01_kr_dyn "${KR_TIME}" "${KR_CPUS}" "${KR_MEM}" "${pc_job}")"

echo "Submitted case01 ${RUN_MODE} workflow"
echo "  RUN_ID: ${RUN_ID}"
echo "  RUN_ROOT: ${RUN_ROOT}"
echo "  resources:"
echo "    replay: ${REPLAY_TIME}, ${REPLAY_CPUS} cpu, ${REPLAY_MEM}"
echo "    Pc:     ${PC_TIME}, ${PC_CPUS} cpu, ${PC_MEM}"
echo "    Kr:     ${KR_TIME}, ${KR_CPUS} cpu, ${KR_MEM}"
echo "  replay job: ${replay_job}"
echo "  Pc job:     ${pc_job}"
echo "  Kr job:     ${kr_job}"
echo
echo "Monitor with:"
echo "  squeue -u ${USER}"
echo "  tail -f ${RUN_ROOT}/slurm/predict_c01_kr_dyn_${kr_job}.out"
