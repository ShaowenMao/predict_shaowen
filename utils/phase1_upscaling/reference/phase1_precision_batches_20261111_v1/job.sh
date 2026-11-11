#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
shift
cd /orcd/data/juanes/001/shaowen/phase1_precision_batches_20261111_v1
/usr/bin/python3.12 -B -m unittest -v test_recovery test_batches
exec /usr/bin/python3.12 -B driver.py "$@"
