#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
cd /orcd/data/juanes/001/shaowen/phase1_solver_targeted_20261110_v1
/usr/bin/python3.12 -B -m unittest test_recovery test_targeted
exec /usr/bin/python3.12 -B driver.py "$2" "$3"
