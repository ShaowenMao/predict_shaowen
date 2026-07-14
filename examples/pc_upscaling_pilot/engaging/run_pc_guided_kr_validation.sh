#!/usr/bin/env bash
# Validate one Swi-medoid Kr result against its completed full-87 benchmark.

set -euo pipefail

: "${KR_VALIDATION_FULL_SUMMARY_CSV:?KR_VALIDATION_FULL_SUMMARY_CSV is required}"
: "${KR_VALIDATION_PROXY_SUMMARY_CSV:?KR_VALIDATION_PROXY_SUMMARY_CSV is required}"
: "${KR_VALIDATION_OUTPUT_DIR:?KR_VALIDATION_OUTPUT_DIR is required}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
REPO_ROOT="${REPO_ROOT:-${PROJECT_ROOT}/repo}"
VALIDATION_DIR="${REPO_ROOT}/examples/pc_upscaling_pilot"
if [[ ! -f "${VALIDATION_DIR}/validate_pc_guided_kr_representatives.m" ]]; then
    echo "Validation code not found: ${VALIDATION_DIR}" >&2
    exit 2
fi
mkdir -p "${KR_VALIDATION_OUTPUT_DIR}"

module load matlab/matlab-2025b
cd "${VALIDATION_DIR}"
matlab -batch "validate_pc_guided_kr_representatives"
