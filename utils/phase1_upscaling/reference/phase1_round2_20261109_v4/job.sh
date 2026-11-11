#!/bin/bash
set -euo pipefail
CODE=/orcd/data/juanes/001/shaowen/phase1_round2_20261109_v4
export PYTHONDONTWRITEBYTECODE=1
export PHASE1_RUN_ROOT="$1"
MODE="$2"
shift 2
cd "$CODE"
case "$MODE" in
    worker) exec /usr/bin/python3.12 -B -u "$CODE/worker.py" "$@" ;;
    recovery) exec /usr/bin/python3.12 -B -u "$CODE/recovery.py" "$@" ;;
    qualify) exec /usr/bin/python3.12 -B -u "$CODE/qualify_timeout.py" ;;
    adopt-window) exec /usr/bin/python3.12 -B -u "$CODE/adopt_window.py" ;;
    *) exit 64 ;;
esac
