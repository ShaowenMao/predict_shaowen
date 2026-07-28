#!/usr/bin/env bash
# Submit the restartable post-QA geology-stratigraphy packaging stage.

set -euo pipefail

ACTION="${1:-plan}"
if [[ "${ACTION}" != "plan" && "${ACTION}" != "submit" ]]; then
    echo "Usage: $0 plan|submit" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7}"
PROJECT_DATA_ROOT="${PROJECT_DATA_ROOT:-/orcd/data/juanes/001/shaowen/predict_shaowen}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
RUN_ID="${RUN_ID:-production_all1620_20260724_v1}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DATA_ROOT}/production_runs/${RUN_ID}}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${RUN_ROOT}/downstream_inputs/geology_stratigraphy}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-mit_amf_advanced_cpu}"
SLURM_QOS="${SLURM_QOS:-mit_amf_advanced_cpu}"
SLURM_PARTITION="${SLURM_PARTITION:-mit_normal}"
PACKAGE_WALLTIME="${PACKAGE_WALLTIME:-08:00:00}"
PACKAGE_MEMORY="${PACKAGE_MEMORY:-16G}"
WORKER="${SCRIPT_DIR}/run_geology_stratigraphy_package.sh"
VERIFY_SCRIPT="${SCRIPT_DIR}/verify_geology_stratigraphy_package.py"

module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64

[[ -d "${RUN_ROOT}" ]] || {
    echo "Missing production run root: ${RUN_ROOT}" >&2
    exit 2
}
[[ -d "${FREEZE_ROOT}/inputs/predict" ]] || {
    echo "Missing frozen PREDICT inputs: ${FREEZE_ROOT}/inputs/predict" >&2
    exit 2
}
[[ -f "${WORKER}" ]] || {
    echo "Missing package worker: ${WORKER}" >&2
    exit 2
}

cat <<EOF
Geology-stratigraphy package plan
  run_id: ${RUN_ID}
  source run: ${RUN_ROOT}
  frozen PREDICT data: ${FREEZE_ROOT}/inputs/predict
  package root: ${PACKAGE_ROOT}
  expected geology MAT files: 162
  expected full-slice fault-case links: 1620
  dependency job: ${DEPENDENCY_JOB_ID:-none; final QA must already be complete}
EOF

if [[ -f "${PACKAGE_ROOT}/geology_stratigraphy.done.json" ]]; then
    python3 "${VERIFY_SCRIPT}" --package-root "${PACKAGE_ROOT}"
    echo "The package is already complete; no submission is needed."
    exit 0
fi

if [[ "${ACTION}" == "plan" ]]; then
    exit 0
fi

if [[ -z "${DEPENDENCY_JOB_ID}" ]]; then
    python3 - "${RUN_ROOT}/case_completion_gate.json" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
if gate.get("status") != "complete" or gate.get("error_count") != 0:
    raise SystemExit("Final case QA must pass before standalone packaging")
PY
fi

log_root="${SCRATCH_ROOT}/production_logs/${RUN_ID}/geology_stratigraphy"
mkdir -p "${log_root}"
dependency_args=()
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
    dependency_args=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
fi

submission="$(
    sbatch \
        --parsable \
        --account="${SLURM_ACCOUNT}" \
        --qos="${SLURM_QOS}" \
        --partition="${SLURM_PARTITION}" \
        --job-name="strat_${RUN_ID}" \
        --time="${PACKAGE_WALLTIME}" \
        --cpus-per-task=1 \
        --mem="${PACKAGE_MEMORY}" \
        "${dependency_args[@]}" \
        --output="${log_root}/%x_%j.out" \
        --error="${log_root}/%x_%j.err" \
        --export=ALL,RUNTIME_REPO="${RUNTIME_REPO}",FREEZE_ROOT="${FREEZE_ROOT}",RUN_ROOT="${RUN_ROOT}",PACKAGE_ROOT="${PACKAGE_ROOT}" \
        "${WORKER}"
)"
job_id="${submission%%;*}"
printf '%s\n' "${job_id}" > "${RUN_ROOT}/geology_stratigraphy_job_id.txt"

python3 - \
    "${RUN_ROOT}/geology_stratigraphy_submission.json" \
    "${RUN_ID}" \
    "${RUN_ROOT}" \
    "${FREEZE_ROOT}" \
    "${PACKAGE_ROOT}" \
    "${DEPENDENCY_JOB_ID}" \
    "${job_id}" <<'PY'
from datetime import datetime, timezone
import json
import sys

(
    output_path,
    run_id,
    run_root,
    freeze_root,
    package_root,
    dependency_job_id,
    job_id,
) = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "status": "submitted",
    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "run_root": run_root,
    "freeze_root": freeze_root,
    "package_root": package_root,
    "expected_geology_count": 162,
    "expected_fault_case_count": 1620,
    "dependency_job_id": dependency_job_id or None,
    "job_id": job_id,
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2)
    stream.write("\n")
print(json.dumps(manifest, indent=2))
PY

echo "Submitted geology-stratigraphy package job ${job_id}."
