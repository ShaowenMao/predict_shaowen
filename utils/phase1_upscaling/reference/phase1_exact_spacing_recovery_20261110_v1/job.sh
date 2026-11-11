#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
shift
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /orcd/data/juanes/001/shaowen/phase1_exact_spacing_recovery_20261110_v1
exec /usr/bin/python3.12 -B -u recover_saved.py "$@"
