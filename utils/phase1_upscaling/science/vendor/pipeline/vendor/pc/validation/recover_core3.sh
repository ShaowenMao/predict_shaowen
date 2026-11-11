#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=02:00:00
#SBATCH --job-name=pc_core3_audit
#SBATCH --no-requeue
set -euo pipefail
module load deprecated-modules gcc/12.2.0-x86_64 python/3.10.8-x86_64 matlab/matlab-2025b julia/1.12.6
: "${JULIA_PC_ROOT:?Set sealed source}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 JULIA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export JULIA_LOAD_PATH="@:@stdlib"
export TMPDIR="/tmp/cpkr_pc_core3_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export MATLAB_PREFDIR="$TMPDIR/pref" MCR_CACHE_ROOT="$TMPDIR/mcr" JULIA_DEPOT_PATH="$TMPDIR/julia:"
mkdir -p "$TMPDIR" "$MATLAB_PREFDIR" "$MCR_CACHE_ROOT"
exec python -B "$JULIA_PC_ROOT/validation/core3_recovery.py" run "$1"
