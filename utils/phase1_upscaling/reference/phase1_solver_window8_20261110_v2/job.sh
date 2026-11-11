#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="$1"
cd /orcd/data/juanes/001/shaowen/phase1_solver_window8_20261110_v2
/usr/bin/python3.12 -B -m unittest test_window
exec /usr/bin/python3.12 -B window.py "${@:2}"
