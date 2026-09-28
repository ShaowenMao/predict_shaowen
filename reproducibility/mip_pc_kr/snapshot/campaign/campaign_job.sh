#!/bin/bash
set -euo pipefail
HERE="$1"
MODE="$2"
export PYTHONDONTWRITEBYTECODE=1
export JULIA_LOAD_PATH='@:@stdlib'
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
if [[ "$MODE" == controller ]]; then
    exec python3.12 -B "$HERE/dispatcher.py" watch
elif [[ "$MODE" == index ]]; then
    exec python3.12 -B "$HERE/index_work.py"
elif [[ "${4:-}" == native ]]; then
    module load deprecated-modules gcc/12.2.0-x86_64 community-modules julia/1.10.4
else
    module load deprecated-modules gcc/12.2.0-x86_64 julia/1.12.6
fi
exec python3.12 -B "$HERE/run_stage.py" "$3" "$4"
