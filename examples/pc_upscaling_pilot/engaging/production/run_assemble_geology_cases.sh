#!/usr/bin/env bash
# Assemble all requested field cases for one geology.

set -euo pipefail

WORK_INDEX="${1:-${SLURM_ARRAY_TASK_ID:-}}"
if [[ -z "${WORK_INDEX}" || ! "${WORK_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 GEOLOGY_WORK_INDEX" >&2
    exit 2
fi

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
CASE_WORK_ROOT="${CASE_WORK_ROOT:?CASE_WORK_ROOT is required}"
CHECKPOINT_OUTPUT_ROOT="${CHECKPOINT_OUTPUT_ROOT:?CHECKPOINT_OUTPUT_ROOT is required}"
CASE_INPUT_ROOT="${CASE_INPUT_ROOT:?CASE_INPUT_ROOT is required}"
WORK_CSV="${CASE_WORK_ROOT}/geology_work.csv"

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
    "geology_work_index",
    "geology_id",
    "scenario_index",
    "scenario_label",
    "case_ids",
    "assignment_count",
    "assignment_relative_path",
    "assignment_sha256",
]
values = [row[field] for field in fields]
if any("\t" in value or "\n" in value for value in values):
    raise SystemExit("Manifest fields must not contain tabs or newlines")
print("\t".join(values))
PY
)"
if [[ -z "${record}" ]]; then
    echo "No geology work item at index ${WORK_INDEX}" >&2
    exit 2
fi
IFS=$'\t' read -r manifest_index geology_id scenario_index scenario_label case_ids \
    assignment_count assignment_relative_path assignment_sha256 <<< "${record}"
if [[ "${manifest_index}" != "${WORK_INDEX}" ]]; then
    echo "Manifest index mismatch: ${manifest_index} != ${WORK_INDEX}" >&2
    exit 2
fi

done_marker="${CASE_INPUT_ROOT}/cases/${geology_id}/geology_case_inputs.done.json"
if [[ -f "${done_marker}" ]]; then
    echo "Geology case inputs already complete: ${geology_id}"
    exit 0
fi

python3 \
    "${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/assemble_geology_case_inputs.py" \
    --geology-id "${geology_id}" \
    --geology-assignment-csv "${CASE_WORK_ROOT}/${assignment_relative_path}" \
    --checkpoint-output-root "${CHECKPOINT_OUTPUT_ROOT}" \
    --output-root "${CASE_INPUT_ROOT}" \
    --case-ids "${case_ids}" \
    --overwrite
