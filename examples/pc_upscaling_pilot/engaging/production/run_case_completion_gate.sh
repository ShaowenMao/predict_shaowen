#!/usr/bin/env bash
# Validate every assembled input and published Pc/Kr result in one run.

set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
MAX_SOURCE_LOG_PERMEABILITY_MISMATCH="${MAX_SOURCE_LOG_PERMEABILITY_MISMATCH:-1.0e-3}"
CASE_WORK_ROOT="${CASE_WORK_ROOT:-${RUN_ROOT}/case_work_manifest}"
CASE_INPUT_ROOT="${CASE_INPUT_ROOT:-${RUN_ROOT}/case_inputs}"
CASE_RESULT_ROOT="${CASE_RESULT_ROOT:-${RUN_ROOT}/case_results}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/case_completion_gate.json}"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/verify_case_completion.py" \
    --case-work-csv "${CASE_WORK_ROOT}/case_work.csv" \
    --case-input-root "${CASE_INPUT_ROOT}" \
    --case-result-root "${CASE_RESULT_ROOT}" \
    --expected-physics-commit "${PHYSICS_COMMIT}" \
    --expected-method-hash "${METHOD_CONFIG_SHA256}" \
    --max-source-log-permeability-mismatch \
    "${MAX_SOURCE_LOG_PERMEABILITY_MISMATCH}" \
    --output-json "${OUTPUT_JSON}"
