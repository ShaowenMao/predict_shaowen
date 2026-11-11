#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="${1:?isolated run root}"
shift
# New submissions pin their runner; shared RUN routing cannot change them.
CODE="/orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_runner_20261030_v5"
mode="${1:?mode}"
shift
case "$mode" in
transition) exec /usr/bin/python3.12 -B "$CODE/transition_operator.py" "$@" ;;
integration) exec /usr/bin/python3.12 -B "$CODE/packet_qualification.py" "$@" ;;
worker) exec /usr/bin/python3.12 -B "$CODE/worker.py" "$@" ;;
control) exec /usr/bin/python3.12 -B "$CODE/control.py" "$@" ;;
fixture) exec /usr/bin/python3.12 -B -c 'import sys;sys.path.insert(0,sys.argv[1]);import control;control.fixture(sys.argv[2])' "$CODE" "$@" ;;
tests) cd "$CODE"; exec /usr/bin/python3.12 -B -m unittest test_runner -v ;;
*) echo "Unknown bounded qualification mode" >&2; exit 2 ;;
esac
