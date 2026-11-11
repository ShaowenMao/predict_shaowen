#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
shift
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /orcd/data/juanes/001/shaowen/phase1_saved16_closeout_20261110_v2
exec /usr/bin/python3.12 -B -u closeout.py "$@"
