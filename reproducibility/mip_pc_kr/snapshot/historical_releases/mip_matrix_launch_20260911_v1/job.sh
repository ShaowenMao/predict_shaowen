#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=18G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${MIP_MATRIX_SOURCE:?}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/orcd/data/juanes/001/shaowen/t/m_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export MATLAB_PREFDIR="$TMPDIR/p" MCR_CACHE_ROOT="$TMPDIR/mcr"
export JULIA_DEPOT_PATH="$TMPDIR/julia:"
mkdir -p "$TMPDIR" "$MATLAB_PREFDIR" "$MCR_CACHE_ROOT"
test "$(realpath "$TMPDIR")" = "$TMPDIR"
export MATRIX_PYTHON=/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/hybrid_pc60_production_kr_s03_c022_case12_20260911_v1/venv/bin/python
"$MATRIX_PYTHON" -B "$MIP_MATRIX_SOURCE/study.py" "$@"
