#!/usr/bin/env bash
# Shared Engaging environment for the median-sand case01 smoke/full tests.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"

RUN_MODE="${RUN_MODE:-smoke}"
RUN_ID="${RUN_ID:-case01_${RUN_MODE}_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/runs/${RUN_ID}}"

mkdir -p "${RUN_ROOT}"/{logs,slurm,tmp,status}

module load matlab/matlab-2025b

export PROJECT_ROOT
export SCRATCH_ROOT
export RUN_MODE
export RUN_ID
export RUN_ROOT
export REPO_ROOT

export MRST_ROOT="${MRST_ROOT:-${PROJECT_ROOT}/software/mrst-current}"
export UPSCALING_ZIP="${UPSCALING_ZIP:-${PROJECT_ROOT}/software/upscaling.zip}"

export FULL87_REPLAY_OUTPUT_ROOT="${FULL87_REPLAY_OUTPUT_ROOT:-${RUN_ROOT}/full87_replay}"
export FULL87_REPLAY_FIELD_SAMPLING_CSV="${FULL87_REPLAY_FIELD_SAMPLING_CSV:-${PROJECT_ROOT}/inputs/texas_offshore_field_sampling/texas_field_slice_window_values.csv}"
export FULL87_REPLAY_DATA_ROOT="${FULL87_REPLAY_DATA_ROOT:-${REPO_ROOT}/examples/thickness_scenario_data}"
export PREDICT_REPLAY_CODE_ROOT="${PREDICT_REPLAY_CODE_ROOT:-${REPO_ROOT}}"
export FULL87_REPLAY_GEOLOGY_ID="${FULL87_REPLAY_GEOLOGY_ID:-s05_c012}"
export FULL87_REPLAY_CASE_IDS="${FULL87_REPLAY_CASE_IDS:-1}"

export PC_IP_GEOLOGY_ID="${PC_IP_GEOLOGY_ID:-${FULL87_REPLAY_GEOLOGY_ID}}"
export PC_IP_CASE_IDS="${PC_IP_CASE_IDS:-${FULL87_REPLAY_CASE_IDS}}"
export PC_IP_REPLAY_ROOT="${PC_IP_REPLAY_ROOT:-${FULL87_REPLAY_OUTPUT_ROOT}}"
export PC_IP_OUTPUT_ROOT="${PC_IP_OUTPUT_ROOT:-${RUN_ROOT}/pc_ip}"

export KR_DYN_GEOLOGY_ID="${KR_DYN_GEOLOGY_ID:-${FULL87_REPLAY_GEOLOGY_ID}}"
export KR_DYN_CASE_IDS="${KR_DYN_CASE_IDS:-${FULL87_REPLAY_CASE_IDS}}"
export KR_DYN_REPLAY_ROOT="${KR_DYN_REPLAY_ROOT:-${FULL87_REPLAY_OUTPUT_ROOT}}"
export KR_DYN_OUTPUT_ROOT="${KR_DYN_OUTPUT_ROOT:-${RUN_ROOT}/kr_dyn}"
export KR_DYN_1D_METHOD="${KR_DYN_1D_METHOD:-ad}"

if [[ "${RUN_MODE}" == "smoke" ]]; then
    export FULL87_REPLAY_MAX_ROWS="${FULL87_REPLAY_MAX_ROWS:-6}"
    export PC_IP_ALLOW_PARTIAL_REPLAY="${PC_IP_ALLOW_PARTIAL_REPLAY:-1}"
    export PC_IP_MAX_ROWS="${PC_IP_MAX_ROWS:-2}"
    export KR_DYN_ALLOW_PARTIAL_REPLAY="${KR_DYN_ALLOW_PARTIAL_REPLAY:-1}"
    export KR_DYN_MAX_ROWS="${KR_DYN_MAX_ROWS:-1}"
    export KR_DYN_USE_PARALLEL="${KR_DYN_USE_PARALLEL:-0}"
    export KR_DYN_3D_NUM_THREADS="${KR_DYN_3D_NUM_THREADS:-2}"
    export KR_DYN_1D_AD_SOLVER="${KR_DYN_1D_AD_SOLVER:-legacy}"
    export KR_DYN_LINEAR_SOLVER="${KR_DYN_LINEAR_SOLVER:-amgcl_require}"
    export KR_DYN_1D_LINEAR_SOLVER="${KR_DYN_1D_LINEAR_SOLVER:-amgcl_auto}"
    export KR_DYN_TIMESTEP_MODE="${KR_DYN_TIMESTEP_MODE:-smoke}"
    export KR_DYN_SMOKE_CARTDIMS="${KR_DYN_SMOKE_CARTDIMS:-20,4,20}"
    export KR_DYN_COREY_STEP="${KR_DYN_COREY_STEP:-3}"
else
    export FULL87_REPLAY_MAX_ROWS="${FULL87_REPLAY_MAX_ROWS:-Inf}"
    export PC_IP_ALLOW_PARTIAL_REPLAY="${PC_IP_ALLOW_PARTIAL_REPLAY:-0}"
    export PC_IP_MAX_ROWS="${PC_IP_MAX_ROWS:-}"
    export KR_DYN_ALLOW_PARTIAL_REPLAY="${KR_DYN_ALLOW_PARTIAL_REPLAY:-0}"
    export KR_DYN_MAX_ROWS="${KR_DYN_MAX_ROWS:-}"
    export KR_DYN_USE_PARALLEL="${KR_DYN_USE_PARALLEL:-1}"
    export KR_DYN_NUM_WORKERS="${KR_DYN_NUM_WORKERS:-12}"
    export KR_DYN_PC_PRESTEP_MODE="${KR_DYN_PC_PRESTEP_MODE:-precomputed}"
    export KR_DYN_3D_NUM_THREADS="${KR_DYN_3D_NUM_THREADS:-1}"
    export KR_DYN_1D_AD_SOLVER="${KR_DYN_1D_AD_SOLVER:-robust}"
    export KR_DYN_LINEAR_SOLVER="${KR_DYN_LINEAR_SOLVER:-amgcl_require}"
    export KR_DYN_1D_LINEAR_SOLVER="${KR_DYN_1D_LINEAR_SOLVER:-amgcl_auto}"
    export KR_DYN_SMOKE_CARTDIMS="${KR_DYN_SMOKE_CARTDIMS:-}"
    export KR_DYN_TIMESTEP_MODE="${KR_DYN_TIMESTEP_MODE:-paper}"
    export KR_DYN_COREY_STEP="${KR_DYN_COREY_STEP:-0.2}"
fi

echo "RUN_ID=${RUN_ID}"
echo "RUN_MODE=${RUN_MODE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "REPO_ROOT=${REPO_ROOT}"
echo "MRST_ROOT=${MRST_ROOT}"
echo "UPSCALING_ZIP=${UPSCALING_ZIP}"
echo "KR_DYN_LINEAR_SOLVER=${KR_DYN_LINEAR_SOLVER:-}"
echo "KR_DYN_1D_LINEAR_SOLVER=${KR_DYN_1D_LINEAR_SOLVER:-}"
