#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=threecase_pc_kr
#SBATCH --no-requeue
#SBATCH --exclude=node1614
set -euo pipefail
module load deprecated-modules gcc/12.2.0-x86_64 matlab/matlab-2025b julia/1.12.6
source_dir=$(cd "$(dirname "$0")" && pwd)
: "${COMPARISON_SOURCE:?Set immutable comparison source}"
study=/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/comparative_3case_wrr30_60_pc_conventions_20260910_v1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 JULIA_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export JULIA_LOAD_PATH='@:@stdlib'
export TMPDIR="$study/tmp/job${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export MATLAB_PREFDIR="$TMPDIR/pref" MCR_CACHE_ROOT="$TMPDIR/mcr" JULIA_DEPOT_PATH="$TMPDIR/julia:"
mkdir -p "$TMPDIR" "$MATLAB_PREFDIR" "$MCR_CACHE_ROOT"
cd "$COMPARISON_SOURCE"
if [[ "$1" == pilot || "$1" == main ]]; then
    exec /usr/bin/python3.12 -B study.py "$1" "$2" "${3:-${SLURM_ARRAY_TASK_ID:?}}"
else
    exec /usr/bin/python3.12 -B study.py "$@"
fi
