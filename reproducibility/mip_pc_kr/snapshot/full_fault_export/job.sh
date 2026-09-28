#!/bin/bash
set -euo pipefail
ROOT="$1"
STAGE="$2"
module load deprecated-modules gcc/12.2.0-x86_64 julia/1.12.6
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
if [[ "$STAGE" == controller ]]; then
    exec python3.12 -B "$ROOT/dispatch.py" watch
else
    exec python3.12 -B "$ROOT/stages.py" "$STAGE"
fi
