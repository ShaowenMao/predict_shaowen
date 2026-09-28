#!/bin/bash
set -euo pipefail
ROOT="$1"
STAGE="$2"
if [[ "$STAGE" == fullgrid || "$STAGE" == probe-reservoir ]]; then
    module load deprecated-modules gcc/12.2.0-x86_64 community-modules julia/1.10.4
else
    module load deprecated-modules gcc/12.2.0-x86_64 julia/1.12.6
fi
export PYTHONDONTWRITEBYTECODE=1
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export JULIA_LOAD_PATH='@:@stdlib'
if [[ "$STAGE" == controller ]]; then
    exec python3.12 -B "$ROOT/qualified_dispatch.py" watch
fi
exec python3.12 -B "$ROOT/qualified_driver.py" "$STAGE" "$(command -v julia)"
