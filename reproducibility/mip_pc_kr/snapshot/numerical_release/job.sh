#!/bin/bash
set -euo pipefail
umask 077
HERE=$1
cd "$HERE"
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64 julia/1.12.6
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 JULIA_NUM_THREADS=2 JULIA_NUM_GC_THREADS=1
if [ "$2" = controller ] || [ "$2" = reuse-index ]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
fi
export JULIA_LOAD_PATH='@:@stdlib'
export TMPDIR="/orcd/data/juanes/001/shaowen/predict_shaowen/campaigns/wrr60_phase1_2430_20260914_v1/numerical_wave01_reuse_v1/tmp/job${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
PY=/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/wrr60_phase1_production_candidate_20260912_v1/reference_integration_v1/integration_v1/environments/export_py310_v1/bin/python
"$PY" -B -m unittest -v test_policy test_campaign test_reuse test_reuse_campaign
exec "$PY" -u -B campaign.py "${@:2}"
