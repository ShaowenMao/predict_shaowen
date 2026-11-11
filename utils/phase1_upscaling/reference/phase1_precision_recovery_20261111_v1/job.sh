#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
shift
cd /orcd/data/juanes/001/shaowen/phase1_precision_recovery_20261111_v1
/usr/bin/python3.12 -B -m unittest -v test_recovery
exec /usr/bin/python3.12 -B driver.py "$@"
