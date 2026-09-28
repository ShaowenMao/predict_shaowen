#!/bin/bash
set -euo pipefail
umask 077
src="$1"; shift
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64 matlab/matlab-2025b
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec srun --cpu-bind=cores /usr/bin/python3.12 "$src/launch.py" "$@"
