#!/bin/bash
set -euo pipefail
exec python3.12 -B "$1/reconcile.py"
