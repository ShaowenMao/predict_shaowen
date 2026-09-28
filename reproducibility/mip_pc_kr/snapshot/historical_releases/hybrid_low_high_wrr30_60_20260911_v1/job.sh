#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${HYBRID_SOURCE:?}"
: "${HYBRID_ROOT:?}"
: "${HYBRID_CASE:?}"
: "${HYBRID_ANGLE:?}"
module purge
module load deprecated-modules gcc/12.2.0-x86_64 matlab/matlab-2025b
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="/orcd/data/juanes/001/shaowen/t/h_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export MATLAB_PREFDIR="$TMPDIR/p" MCR_CACHE_ROOT="$TMPDIR/mcr"
export JULIA_DEPOT_PATH="$TMPDIR/julia:"
test "${#TMPDIR}" -lt 50
mkdir -p "$TMPDIR" "$MATLAB_PREFDIR" "$MCR_CACHE_ROOT"
test "$(realpath "$TMPDIR")" = "$TMPDIR"
cd "$HYBRID_ROOT"
if [[ "$1" = prepare && ! -x "$HYBRID_ROOT/venv/bin/python" ]]; then
    /usr/bin/python3.12 -m venv "$HYBRID_ROOT/venv"
    "$HYBRID_ROOT/venv/bin/python" -m pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 h5py==3.14.0
    "$HYBRID_ROOT/venv/bin/python" -m pip freeze > "$HYBRID_ROOT/python_dependencies.txt"
fi
"$HYBRID_ROOT/venv/bin/python" -B -m unittest discover -s "$HYBRID_SOURCE" -p 'test_*.py'
"$HYBRID_ROOT/venv/bin/python" -B "$HYBRID_SOURCE/study.py" "$1" "${2:-${SLURM_ARRAY_TASK_ID:-0}}"
