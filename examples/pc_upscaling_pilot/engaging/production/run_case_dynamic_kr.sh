#!/usr/bin/env bash
# Replay six Swi-medoid representatives and run one rigorous dynamic-Kr case.

set -euo pipefail

WORK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${WORK_INDEX}" || ! "${WORK_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 CASE_WORK_INDEX" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
FROZEN_REPO="${FROZEN_REPO:-${FREEZE_ROOT}/code/source}"
CASE_WORK_ROOT="${CASE_WORK_ROOT:?CASE_WORK_ROOT is required}"
CASE_INPUT_ROOT="${CASE_INPUT_ROOT:?CASE_INPUT_ROOT is required}"
CASE_RESULT_ROOT="${CASE_RESULT_ROOT:?CASE_RESULT_ROOT is required}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
MRST_ROOT="${MRST_ROOT:-${PROJECT_ROOT}/software/mrst-current}"
UPSCALING_ZIP="${UPSCALING_ZIP:-${PROJECT_ROOT}/software/upscaling.zip}"
PREDICT_ROOT="${PREDICT_ROOT:-${FREEZE_ROOT}/inputs/predict}"
PERMEABILITY_INPUT="${PERMEABILITY_INPUT:-${FREEZE_ROOT}/inputs/sampling/texas_field_sampling_compact.mat}"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-1.0e-3}"
WORK_CSV="${CASE_WORK_ROOT}/case_work.csv"

module load deprecated-modules gcc/12.2.0-x86_64 \
    python/3.10.8-x86_64 matlab/matlab-2025b

record="$(
    python3 - "${WORK_CSV}" "${WORK_INDEX}" <<'PY'
import csv
import sys

work_csv = sys.argv[1]
work_index = int(sys.argv[2])
with open(work_csv, newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
if work_index < 1 or work_index > len(rows):
    raise SystemExit(0)
row = rows[work_index - 1]
fields = [
    "case_work_index",
    "geology_id",
    "scenario_index",
    "scenario_label",
    "case_id",
    "case_name",
    "case_category",
    "case_relative_path",
]
values = [row[field] for field in fields]
if any("\t" in value or "\n" in value for value in values):
    raise SystemExit("Manifest fields must not contain tabs or newlines")
print("\t".join(values))
PY
)"
if [[ -z "${record}" ]]; then
    echo "No case work item at index ${WORK_INDEX}" >&2
    exit 2
fi
IFS=$'\t' read -r manifest_index geology_id scenario_index scenario_label case_id \
    case_name case_category case_relative_path <<< "${record}"
if [[ "${manifest_index}" != "${WORK_INDEX}" ]]; then
    echo "Manifest index mismatch: ${manifest_index} != ${WORK_INDEX}" >&2
    exit 2
fi
printf -v case_two_digit '%02d' "${case_id}"

INPUT_DIR="${CASE_INPUT_ROOT}/${case_relative_path}/inputs"
OUTPUT_DIR="${CASE_RESULT_ROOT}/${case_relative_path}"
DONE_MARKER="${OUTPUT_DIR}/case.done.json"
if [[ -f "${DONE_MARKER}" ]]; then
    echo "Dynamic Kr case already complete: ${geology_id} case ${case_two_digit}"
    exit 0
fi

shopt -s nullglob
representative_selection=("${INPUT_DIR}"/kr_representative_replay_*.csv)
pc_summary=("${INPUT_DIR}"/pc_curve_summary_*_ip_full87.csv)
pc_fixed=("${INPUT_DIR}"/pc_curve_points_*_ip_full87.csv)
pc_native=("${INPUT_DIR}"/pc_native_curve_points_*_ip_full87.csv)
replay_template=("${INPUT_DIR}"/replay_summary_template_*.csv)
[[ "${#representative_selection[@]}" -eq 1 ]] || {
    echo "Expected one representative selection in ${INPUT_DIR}." >&2; exit 2; }
[[ "${#pc_summary[@]}" -eq 1 ]] || {
    echo "Expected one Pc summary in ${INPUT_DIR}." >&2; exit 2; }
[[ "${#pc_fixed[@]}" -eq 1 ]] || {
    echo "Expected one fixed-grid Pc table in ${INPUT_DIR}." >&2; exit 2; }
[[ "${#pc_native[@]}" -eq 1 ]] || {
    echo "Expected one native Pc table in ${INPUT_DIR}." >&2; exit 2; }
[[ "${#replay_template[@]}" -eq 1 ]] || {
    echo "Expected one replay template in ${INPUT_DIR}." >&2; exit 2; }

JOB_TOKEN="${SLURM_JOB_ID:-manual}_${WORK_INDEX}"
LOCAL_BASE="${SLURM_TMPDIR:-${TMPDIR:-${SCRATCH_ROOT}/tmp}}"
LOCAL_ROOT="${LOCAL_BASE}/predict_case_${JOB_TOKEN}"
REPLAY_ROOT="${LOCAL_ROOT}/representative_replay"
KR_ROOT="${LOCAL_ROOT}/kr"
UPSCALING_ROOT="${LOCAL_ROOT}/upscaling"
mkdir -p "${REPLAY_ROOT}" "${KR_ROOT}" "${UPSCALING_ROOT}"

cleanup() {
    local status="$?"
    if [[ "${KEEP_CASE_TEMP:-0}" != "1" ]]; then
        rm -rf "${LOCAL_ROOT}"
    else
        echo "Retaining case temporary files: ${LOCAL_ROOT}" >&2
    fi
    exit "${status}"
}
trap cleanup EXIT

echo "geology_id=${geology_id}"
echo "case_id=${case_two_digit}"
echo "case_name=${case_name}"
echo "hostname=$(hostname)"
echo "started_at=$(date --iso-8601=seconds)"

matlab -batch \
    "addpath('${RUNTIME_REPO}/examples/pc_upscaling_pilot'); prepare_production_replay_batch('${representative_selection[0]}', '${REPLAY_ROOT}', '${PREDICT_ROOT}', '${FROZEN_REPO}', '${MRST_ROOT}', ${REPLAY_TOLERANCE_LOG10});"

merged_replay="${REPLAY_ROOT}/tables/replay_summary_full_case.csv"
python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/merge_representative_replay.py" \
    --template-csv "${replay_template[0]}" \
    --representative-selection-csv "${representative_selection[0]}" \
    --representative-replay-summary-csv "${REPLAY_ROOT}/tables/replay_summary_context.csv" \
    --output-csv "${merged_replay}" \
    --tolerance-log10 "${REPLAY_TOLERANCE_LOG10}"

export KR_DYN_GEOLOGY_ID="${geology_id}"
export KR_DYN_CASE_IDS="${case_id}"
export KR_DYN_REPLAY_ROOT="${REPLAY_ROOT}"
export KR_DYN_REPLAY_SUMMARY_CSV="${merged_replay}"
export KR_DYN_OUTPUT_ROOT="${KR_ROOT}"
export KR_DYN_SELECTION_MODE="swi_medoid"
export KR_DYN_PC_PRESTEP_MODE="precomputed"
export KR_DYN_PRECOMPUTED_PC_SUMMARY_CSV="${pc_summary[0]}"
export KR_DYN_PRECOMPUTED_PC_CURVE_CSV="${pc_fixed[0]}"
export KR_DYN_PRECOMPUTED_PC_NATIVE_CURVE_CSV="${pc_native[0]}"
export KR_DYN_PERMEABILITY_INPUT="${PERMEABILITY_INPUT}"
export KR_DYN_RESERVOIR_PC_REPRESENTATION="both"
export KR_DYN_EXPORT_RESERVOIR_READY="1"
export KR_DYN_ALLOW_PARTIAL_REPLAY="0"
export KR_DYN_MAX_ROWS=""
export KR_DYN_USE_PARALLEL="1"
export KR_DYN_NUM_WORKERS="${SLURM_CPUS_PER_TASK:-6}"
export KR_DYN_3D_NUM_THREADS="1"
export KR_DYN_1D_AD_SOLVER="robust"
export KR_DYN_LINEAR_SOLVER="amgcl_require"
export KR_DYN_1D_LINEAR_SOLVER="amgcl_auto"
export KR_DYN_TIMESTEP_MODE="paper"
export KR_DYN_COREY_STEP="0.2"
export KR_DYN_UPSCALING_ROOT="${UPSCALING_ROOT}"
export MRST_ROOT UPSCALING_ZIP

matlab -batch \
    "run('${FROZEN_REPO}/examples/pc_upscaling_pilot/run_kr_upscaling_dyn_median_examples_full87.m');"

python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/finalize_case_kr.py" \
    --geology-id "${geology_id}" \
    --case-id "${case_id}" \
    --kr-root "${KR_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --physics-commit "${PHYSICS_COMMIT}" \
    --method-config-sha256 "${METHOD_CONFIG_SHA256}" \
    --overwrite-incomplete

echo "finished_at=$(date --iso-8601=seconds)"
echo "Published case result: ${OUTPUT_DIR}"
