#!/bin/bash
set -euo pipefail
export PHASE1_RUN_ROOT="${1:?isolated run root}"
shift
CODE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# Slurm spools launchers; resolve sealed code from the supplied isolated root.
CODE="$(/usr/bin/python3.12 -B -c 'import json,sys; print(json.load(open(sys.argv[1]+"/RUN.json"))["code"])' "$PHASE1_RUN_ROOT")"
mode="${1:?mode}"
shift
case "$mode" in
integration) exec /usr/bin/python3.12 -B "$CODE/packet_qualification.py" "$@" ;;
worker) exec /usr/bin/python3.12 -B "$CODE/worker.py" "$@" ;;
control) exec /usr/bin/python3.12 -B "$CODE/control.py" "$@" ;;
fixture) exec /usr/bin/python3.12 -B -c 'import sys;sys.path.insert(0,sys.argv[1]);import control;control.fixture(sys.argv[2])' "$CODE" "$@" ;;
tests) cd "$CODE"; exec /usr/bin/python3.12 -B -m unittest test_runner -v ;;
*) echo "Unknown bounded qualification mode" >&2; exit 2 ;;
esac
