#!/usr/bin/env bash
# Validate all checkpoint replay/Pc artifacts before releasing case assembly.

set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
DEFAULT_REPLAY_TOLERANCE_LOG10="${DEFAULT_REPLAY_TOLERANCE_LOG10:-1.0e-3}"
REPLAY_TOLERANCE_EXCEPTIONS="${REPLAY_TOLERANCE_EXCEPTIONS:-}"
CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT:-${RUN_ROOT}/checkpoint_manifest}"
CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:-${RUN_ROOT}/checkpoint_pc}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_ROOT}/checkpoint_completion_gate.json}"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

args=(
    --checkpoint-groups-csv "${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv"
    --checkpoint-output-root "${CHECKPOINT_OUTPUT_ROOT}"
    --expected-physics-commit "${PHYSICS_COMMIT}"
    --expected-method-hash "${METHOD_CONFIG_SHA256}"
    --default-replay-tolerance-log10 "${DEFAULT_REPLAY_TOLERANCE_LOG10}"
    --output-json "${OUTPUT_JSON}"
)

if [[ -n "${REPLAY_TOLERANCE_EXCEPTIONS}" ]]; then
    IFS=',' read -r -a exception_items <<< "${REPLAY_TOLERANCE_EXCEPTIONS}"
    for exception_item in "${exception_items[@]}"; do
        args+=(--tolerance-exception "${exception_item}")
    done
fi

python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/verify_checkpoint_completion.py" \
    "${args[@]}"
