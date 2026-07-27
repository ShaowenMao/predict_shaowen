#!/usr/bin/env bash
# Verify that a MATLAB Processes pool can use node-local coordination storage.

set -euo pipefail

WORKER_COUNT="${SLURM_CPUS_PER_TASK:-6}"
NODE_LOCAL_TMP_ROOT="${NODE_LOCAL_TMP_ROOT:-/tmp/${USER}/predict_shaowen}"
JOB_TOKEN="${SLURM_JOB_ID:-manual}_parpool_smoke"
MATLAB_RUNTIME_ROOT="${NODE_LOCAL_TMP_ROOT}/matlab_${JOB_TOKEN}"
MATLAB_JOB_STORAGE="${MATLAB_RUNTIME_ROOT}/local_cluster_jobs"
export TMPDIR="${MATLAB_RUNTIME_ROOT}/t"
export MATLAB_PREFDIR="${MATLAB_RUNTIME_ROOT}/p"

module load deprecated-modules gcc/12.2.0-x86_64 \
    python/3.10.8-x86_64 matlab/matlab-2025b

cleanup() {
    local status="$?"
    rm -rf "${MATLAB_RUNTIME_ROOT}"
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${TMPDIR}" "${MATLAB_PREFDIR}" "${MATLAB_JOB_STORAGE}"
touch "${MATLAB_JOB_STORAGE}/.write_test"
rm -f "${MATLAB_JOB_STORAGE}/.write_test"

echo "hostname=$(hostname)"
echo "worker_count=${WORKER_COUNT}"
echo "matlab_tempdir=${TMPDIR}"
echo "matlab_prefdir=${MATLAB_PREFDIR}"
echo "matlab_job_storage=${MATLAB_JOB_STORAGE}"

matlab -batch \
    "cluster = parcluster('Processes'); cluster.JobStorageLocation = '${MATLAB_JOB_STORAGE}'; saveProfile(cluster); pool = parpool(cluster, ${WORKER_COUNT}); values = zeros(1, ${WORKER_COUNT}); parfor worker = 1:${WORKER_COUNT}; values(worker) = worker + 100; end; assert(isequal(values, 101:(100 + ${WORKER_COUNT}))); delete(pool); fprintf('NODE_LOCAL_PARPOOL_SMOKE_OK workers=%d\\n', ${WORKER_COUNT});"
