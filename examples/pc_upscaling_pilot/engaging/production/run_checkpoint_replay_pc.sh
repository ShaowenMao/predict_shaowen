#!/usr/bin/env bash
# Run one checkpoint-centered replay + Pc work unit.

set -euo pipefail

GROUP_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${GROUP_INDEX}" || ! "${GROUP_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 GROUP_INDEX (or set SLURM_ARRAY_TASK_ID)" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
FROZEN_REPO="${FROZEN_REPO:-${FREEZE_ROOT}/code/source}"
CHECKPOINT_MANIFEST_ROOT="${CHECKPOINT_MANIFEST_ROOT:?CHECKPOINT_MANIFEST_ROOT is required}"
COMPACT_OUTPUT_ROOT="${COMPACT_OUTPUT_ROOT:?COMPACT_OUTPUT_ROOT is required}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/shaowen/orcd/pool/predict_shaowen}"
MRST_ROOT="${MRST_ROOT:-${PROJECT_ROOT}/software/mrst-current}"
UPSCALING_ZIP="${UPSCALING_ZIP:-${PROJECT_ROOT}/software/upscaling.zip}"
PREDICT_ROOT="${PREDICT_ROOT:-${FREEZE_ROOT}/inputs/predict}"
METHOD_CONFIG="${METHOD_CONFIG:-${FREEZE_ROOT}/config/production_method_config.toml}"
PHYSICS_COMMIT="${PHYSICS_COMMIT:-68351e35f3679317b35532a9ca0533674e0aafb5}"
METHOD_CONFIG_SHA256="${METHOD_CONFIG_SHA256:-21266acc83f38d374cdc966d8243834e92b786b75ab1f90dd0a99f4244717a8f}"
REPLAY_TOLERANCE_LOG10="${REPLAY_TOLERANCE_LOG10:-1.0e-3}"

module load deprecated-modules gcc/12.2.0-x86_64 \
    python/3.10.8-x86_64 matlab/matlab-2025b

GROUPS_CSV="${CHECKPOINT_MANIFEST_ROOT}/checkpoint_groups.csv"
[[ -f "${GROUPS_CSV}" ]] || {
    echo "Missing checkpoint group manifest: ${GROUPS_CSV}" >&2
    exit 2
}

record="$(
    python3 - "${GROUPS_CSV}" "${GROUP_INDEX}" <<'PY'
import csv
import sys

groups_csv = sys.argv[1]
group_index = int(sys.argv[2])
with open(groups_csv, newline="", encoding="utf-8-sig") as stream:
    rows = list(csv.DictReader(stream))
if group_index < 1 or group_index > len(rows):
    raise SystemExit(0)
row = rows[group_index - 1]
fields = [
    "group_index",
    "group_id",
    "geology_id",
    "scenario_index",
    "window",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "task_count",
    "usage_count",
    "selection_relative_path",
    "output_relative_path",
    "done_marker_relative_path",
]
values = [row[field] for field in fields]
if any("\t" in value or "\n" in value for value in values):
    raise SystemExit("Manifest fields must not contain tabs or newlines")
print("\t".join(values))
PY
)"
if [[ -z "${record}" ]]; then
    echo "No checkpoint group at index ${GROUP_INDEX}" >&2
    exit 2
fi
IFS=$'\t' read -r manifest_index group_id geology_id scenario_index window_name \
    checkpoint_relative_path checkpoint_sha256 task_count usage_count \
    selection_relative_path output_relative_path done_relative_path <<< "${record}"
if [[ "${manifest_index}" != "${GROUP_INDEX}" ]]; then
    echo "Manifest index mismatch: ${manifest_index} != ${GROUP_INDEX}" >&2
    exit 2
fi

SELECTION_CSV="${CHECKPOINT_MANIFEST_ROOT}/${selection_relative_path}"
OUTPUT_DIR="${COMPACT_OUTPUT_ROOT}/${group_id}"
DONE_MARKER="${OUTPUT_DIR}/checkpoint.done.json"
[[ -f "${SELECTION_CSV}" ]] || {
    echo "Missing checkpoint selection: ${SELECTION_CSV}" >&2
    exit 2
}

if [[ -f "${DONE_MARKER}" ]]; then
    python3 - "${DONE_MARKER}" "${group_id}" "${checkpoint_sha256}" \
        "${PHYSICS_COMMIT}" "${METHOD_CONFIG_SHA256}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
expected = {
    "status": "complete",
    "group_id": sys.argv[2],
    "checkpoint_sha256": sys.argv[3],
    "physics_commit": sys.argv[4],
    "method_config_sha256": sys.argv[5],
}
for key, value in expected.items():
    if data.get(key) != value:
        raise SystemExit(f"Done marker mismatch for {key}: {data.get(key)!r}")
print(f"Checkpoint group already complete: {data['group_id']}")
PY
    exit 0
fi

JOB_TOKEN="${SLURM_JOB_ID:-manual}_${GROUP_INDEX}"
LOCAL_BASE="${SLURM_TMPDIR:-${TMPDIR:-${SCRATCH_ROOT}/tmp}}"
LOCAL_ROOT="${LOCAL_BASE}/predict_checkpoint_${JOB_TOKEN}"
REPLAY_ROOT="${LOCAL_ROOT}/replay"
PC_ROOT="${LOCAL_ROOT}/pc_ip"
UPSCALING_ROOT="${LOCAL_ROOT}/upscaling_empty"
mkdir -p "${REPLAY_ROOT}" "${PC_ROOT}" "${UPSCALING_ROOT}"

cleanup() {
    local status="$?"
    if [[ "${KEEP_CHECKPOINT_TEMP:-0}" != "1" ]]; then
        rm -rf "${LOCAL_ROOT}"
    else
        echo "Retaining checkpoint temporary files: ${LOCAL_ROOT}" >&2
    fi
    exit "${status}"
}
trap cleanup EXIT

echo "group_id=${group_id}"
echo "geology_id=${geology_id}"
echo "window=${window_name}"
echo "task_count=${task_count}"
echo "usage_count=${usage_count}"
echo "hostname=$(hostname)"
echo "local_root=${LOCAL_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "started_at=$(date --iso-8601=seconds)"

matlab -batch \
    "addpath('${RUNTIME_REPO}/examples/pc_upscaling_pilot'); prepare_production_replay_batch('${SELECTION_CSV}', '${REPLAY_ROOT}', '${PREDICT_ROOT}', '${FROZEN_REPO}', '${MRST_ROOT}', ${REPLAY_TOLERANCE_LOG10});"

export PC_IP_GEOLOGY_ID="${geology_id}"
export PC_IP_CASE_IDS="1"
export PC_IP_REPLAY_ROOT="${REPLAY_ROOT}"
export PC_IP_REPLAY_SUMMARY_CSV="${REPLAY_ROOT}/tables/replay_summary_context.csv"
export PC_IP_OUTPUT_ROOT="${PC_ROOT}"
export PC_IP_ALLOW_PARTIAL_REPLAY="1"
export PC_IP_ENABLE_MEDOID_DIAGNOSTICS="0"
export PC_IP_MAX_ROWS=""
export PC_IP_UPSCALING_ROOT="${UPSCALING_ROOT}"
export MRST_ROOT UPSCALING_ZIP

matlab -batch \
    "run('${FROZEN_REPO}/examples/pc_upscaling_pilot/run_pc_upscaling_ip_median_examples_full87.m');"

python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/finalize_checkpoint_pc.py" \
    --group-id "${group_id}" \
    --selection-csv "${SELECTION_CSV}" \
    --replay-summary-csv "${REPLAY_ROOT}/tables/replay_summary_context.csv" \
    --pc-root "${PC_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-sha256 "${checkpoint_sha256}" \
    --physics-commit "${PHYSICS_COMMIT}" \
    --method-config-sha256 "${METHOD_CONFIG_SHA256}" \
    --replay-tolerance-log10 "${REPLAY_TOLERANCE_LOG10}" \
    --overwrite-incomplete

echo "finished_at=$(date --iso-8601=seconds)"
echo "Published compact checkpoint output: ${OUTPUT_DIR}"
