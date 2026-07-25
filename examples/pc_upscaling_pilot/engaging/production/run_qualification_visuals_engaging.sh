#!/usr/bin/env bash
# Render visual QA from an already completed qualification batch.

#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=qual_visuals

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
CODE_ROOT="${CODE_ROOT:-${PROJECT_ROOT}/examples/pc_upscaling_pilot/engaging/production}"
BATCH_ROOT="${BATCH_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/qualification_results/qualification_ccu_20260722_v3b_fullgrid}"
OUTPUT_DIR="${OUTPUT_DIR:-/orcd/data/juanes/001/shaowen/predict_shaowen/qualification_visuals/qualification_ccu_20260722_v3b_fullgrid}"
REPRESENTATIVE_GEOLOGY="${REPRESENTATIVE_GEOLOGY:-s05_c012}"

[[ -f "${CODE_ROOT}/plot_qualification_batch_visuals.m" ]] || {
    echo "Missing plotting function: ${CODE_ROOT}/plot_qualification_batch_visuals.m" >&2
    exit 2
}
[[ -d "${BATCH_ROOT}/cases" ]] || {
    echo "Missing qualification cases folder: ${BATCH_ROOT}/cases" >&2
    exit 2
}

mkdir -p "${OUTPUT_DIR}"
module load matlab/matlab-2025b

matlab -batch "addpath('${CODE_ROOT}'); plot_qualification_batch_visuals('${BATCH_ROOT}', '${OUTPUT_DIR}', 'RepresentativeGeology', '${REPRESENTATIVE_GEOLOGY}');"

echo "Qualification visual QA complete: ${OUTPUT_DIR}"
