#!/usr/bin/env bash
# Build and submit checkpoint-centered replay/Pc work units on Engaging.

set -euo pipefail

MODE="${1:-}"
if [[ "${MODE}" != "qualification60" && "${MODE}" != "full" ]]; then
    echo "Usage: $0 qualification60|full" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
PREDICT_ROOT="${PREDICT_ROOT:-${FREEZE_ROOT}/inputs/predict}"
SAMPLING_ROOT="${SAMPLING_ROOT:-${FREEZE_ROOT}/inputs/sampling}"
SAMPLING_CSV="${SAMPLING_ROOT}/texas_field_slice_window_values.csv"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"

if [[ "${MODE}" == "qualification60" ]]; then
    RUN_ID="${RUN_ID:-production_qualification60_20260723_v1}"
    CASE_FILTER="${CASE_FILTER:-${SCRIPT_DIR}/production_qualification_60_cases.csv}"
    MAX_CONCURRENT="${MAX_CONCURRENT:-18}"
else
    RUN_ID="${RUN_ID:-production_all1620_20260723_v1}"
    CASE_FILTER=""
    MAX_CONCURRENT="${MAX_CONCURRENT:-96}"
fi

RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT:-${RUN_ROOT}/checkpoint_manifest}"
COMPACT_OUTPUT_ROOT="${COMPACT_OUTPUT_ROOT:-${RUN_ROOT}/checkpoint_pc}"
LOG_ROOT="${LOG_ROOT:-${SCRATCH_ROOT}/production_logs/${RUN_ID}/checkpoint_pc}"
SEMANTIC_PREFLIGHT="${RUN_ROOT}/semantic_preflight.json"
WORKER="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/run_checkpoint_replay_pc.sh"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

preflight_args=(
    --freeze-root "${FREEZE_ROOT}"
    --predict-root "${PREDICT_ROOT}"
    --sampling-root "${SAMPLING_ROOT}"
    --expected-code-commit "${PHYSICS_COMMIT}"
    --expected-method-hash "${METHOD_CONFIG_SHA256}"
    --output-json "${SEMANTIC_PREFLIGHT}"
)
python3 "${SCRIPT_DIR}/verify_production_freeze.py" "${preflight_args[@]}"

if [[ ! -f "${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv" ]]; then
    build_args=(
        --freeze-root "${FREEZE_ROOT}"
        --predict-root "${PREDICT_ROOT}"
        --sampling-csv "${SAMPLING_CSV}"
        --output-root "${CHECKPOINT_MANIFEST_ROOT}"
    )
    if [[ -n "${CASE_FILTER}" ]]; then
        build_args+=(--case-filter-csv "${CASE_FILTER}")
    fi
    python3 "${SCRIPT_DIR}/build_checkpoint_work_manifest.py" "${build_args[@]}"
fi

GROUP_COUNT="$(
    python3 - "${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as stream:
    print(sum(1 for _ in csv.DictReader(stream)))
PY
)"
if [[ "${GROUP_COUNT}" -le 0 ]]; then
    echo "Checkpoint manifest contains no work units." >&2
    exit 2
fi

cat > "${RUN_ROOT}/checkpoint_submission.env" <<EOF
run_id=${RUN_ID}
mode=${MODE}
created_at=$(date --iso-8601=seconds)
freeze_root=${FREEZE_ROOT}
physics_commit=${PHYSICS_COMMIT}
method_config_sha256=${METHOD_CONFIG_SHA256}
checkpoint_manifest_root=${CHECKPOINT_MANIFEST_ROOT}
compact_output_root=${COMPACT_OUTPUT_ROOT}
group_count=${GROUP_COUNT}
max_concurrent=${MAX_CONCURRENT}
slurm_qos=${SLURM_QOS}
EOF

submission="$(
    sbatch \
        --parsable \
        --account=mit_amf_advanced_cpu \
        --qos="${SLURM_QOS}" \
        --partition=mit_normal \
        --job-name="rpc_${RUN_ID}" \
        --time="${CHECKPOINT_WALLTIME:-04:00:00}" \
        --cpus-per-task=1 \
        --mem="${CHECKPOINT_MEMORY:-16G}" \
        --array="1-${GROUP_COUNT}%${MAX_CONCURRENT}" \
        --output="${LOG_ROOT}/%x_%A_%a.out" \
        --error="${LOG_ROOT}/%x_%A_%a.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",FREEZE_ROOT="${FREEZE_ROOT}",CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT}",COMPACT_OUTPUT_ROOT="${COMPACT_OUTPUT_ROOT}",SCRATCH_ROOT="${SCRATCH_ROOT}",PHYSICS_COMMIT="${PHYSICS_COMMIT}",METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256}" \
        "${WORKER}"
)"
JOB_ID="${submission%%;*}"
echo "${JOB_ID}" > "${RUN_ROOT}/checkpoint_array_job_id.txt"
echo "Submitted ${GROUP_COUNT} checkpoint replay/Pc work units: job ${JOB_ID}"
echo "Run root: ${RUN_ROOT}"
