#!/usr/bin/env bash
# Build and atomically publish the 162-geology stratigraphy companion package.

set -euo pipefail

RUNTIME_REPO="${RUNTIME_REPO:-/home/shaowen/orcd/pool/predict_shaowen}"
FREEZE_ROOT="${FREEZE_ROOT:?FREEZE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PREDICT_DATA_ROOT="${PREDICT_DATA_ROOT:-${FREEZE_ROOT}/inputs/predict}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${RUN_ROOT}/downstream_inputs/geology_stratigraphy}"
OVERWRITE_INCOMPLETE="${OVERWRITE_INCOMPLETE:-0}"
VERIFY_SCRIPT="${RUNTIME_REPO}/examples/pc_upscaling_pilot/engaging/production/verify_geology_stratigraphy_package.py"
MATLAB_SOURCE_ROOT="${RUNTIME_REPO}/examples/pc_upscaling_pilot"
COMPLETION_GATE="${RUN_ROOT}/case_completion_gate.json"

module load deprecated-modules gcc/12.2.0-x86_64 \
    python/3.10.8-x86_64 matlab/matlab-2025b

[[ -f "${COMPLETION_GATE}" ]] || {
    echo "Missing final case-completion gate: ${COMPLETION_GATE}" >&2
    exit 2
}
[[ -d "${PREDICT_DATA_ROOT}" ]] || {
    echo "Missing frozen PREDICT data root: ${PREDICT_DATA_ROOT}" >&2
    exit 2
}
[[ -f "${VERIFY_SCRIPT}" ]] || {
    echo "Missing package verifier: ${VERIFY_SCRIPT}" >&2
    exit 2
}

python3 - "${COMPLETION_GATE}" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {
    "status": "complete",
    "expected_geology_count": 162,
    "expected_case_count": 1620,
    "result_markers_validated": 1620,
    "error_count": 0,
}
for field, value in expected.items():
    if gate.get(field) != value:
        raise SystemExit(
            f"Final case gate {field}={gate.get(field)!r}; expected {value!r}"
        )
PY

if [[ -f "${PACKAGE_ROOT}/geology_stratigraphy.done.json" ]]; then
    python3 "${VERIFY_SCRIPT}" --package-root "${PACKAGE_ROOT}"
    echo "Geology-stratigraphy package is already complete: ${PACKAGE_ROOT}"
    exit 0
fi

if [[ -e "${PACKAGE_ROOT}" ]]; then
    if [[ "${OVERWRITE_INCOMPLETE}" != "1" ]]; then
        echo "Incomplete package exists; set OVERWRITE_INCOMPLETE=1: ${PACKAGE_ROOT}" >&2
        exit 2
    fi
    python3 - "${RUN_ROOT}" "${PACKAGE_ROOT}" <<'PY'
import shutil
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
package_root = Path(sys.argv[2]).resolve()
if run_root not in package_root.parents:
    raise SystemExit(f"Refusing to remove path outside run root: {package_root}")
shutil.rmtree(package_root)
PY
fi

job_token="${SLURM_JOB_ID:-manual_$(date +%Y%m%dT%H%M%S)}"
package_parent="$(dirname "${PACKAGE_ROOT}")"
staging_root="${package_parent}/.geology_stratigraphy.partial.${job_token}"
mkdir -p "${package_parent}"
if [[ -e "${staging_root}" ]]; then
    echo "Staging path already exists: ${staging_root}" >&2
    exit 2
fi

node_local_root="/tmp/${USER}/predict_shaowen/geology_stratigraphy_${job_token}"
export TMPDIR="${node_local_root}/tmp"
export MATLAB_PREFDIR="${node_local_root}/preferences"
export PREDICT_DATA_ROOT RUN_ROOT
export STRATIGRAPHY_STAGING_ROOT="${staging_root}"
mkdir -p "${TMPDIR}" "${MATLAB_PREFDIR}"

cleanup() {
    local status="$?"
    if [[ "${status}" -ne 0 ]]; then
        rm -rf "${staging_root}"
    fi
    rm -rf "${node_local_root}"
    exit "${status}"
}
trap cleanup EXIT

matlab -batch "addpath('${MATLAB_SOURCE_ROOT}'); build_production_geology_stratigraphy_package(getenv('PREDICT_DATA_ROOT'), getenv('RUN_ROOT'), getenv('STRATIGRAPHY_STAGING_ROOT'));"

python3 "${VERIFY_SCRIPT}" --package-root "${staging_root}"

python3 - "${staging_root}" "${PACKAGE_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

staging = Path(sys.argv[1])
target = Path(sys.argv[2])
if target.exists():
    raise SystemExit(f"Refusing to replace existing package: {target}")
os.replace(staging, target)
PY

python3 "${VERIFY_SCRIPT}" --package-root "${PACKAGE_ROOT}"
printf '%s\n' "${PACKAGE_ROOT}" > "${RUN_ROOT}/geology_stratigraphy_package_path.txt"
echo "Published geology-stratigraphy package: ${PACKAGE_ROOT}"
