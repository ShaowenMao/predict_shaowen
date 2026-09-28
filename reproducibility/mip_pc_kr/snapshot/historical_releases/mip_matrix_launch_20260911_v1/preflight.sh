#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
: "${MIP_MATRIX_SOURCE:?}"
export STUDY_ROOT="$1" GOM_GRID_ROOT="$1"
export JUTULDARCY_COMBINED_REPO=/home/shaowen/orcd/scratch/gom_grid/code/immutable/JutulDarcy_ed81af8e
export TMPDIR="/orcd/data/juanes/001/shaowen/t/m_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/hybrid_pc60_production_kr_s03_c022_case12_20260911_v1/venv/bin/python -B "$MIP_MATRIX_SOURCE/study.py" qualify "$STUDY_ROOT"
exec bash "$STUDY_ROOT/source/gom_step62_effective_pc_global_plateau_preflight.sbatch"
