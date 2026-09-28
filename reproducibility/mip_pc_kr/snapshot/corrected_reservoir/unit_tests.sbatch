#!/bin/bash
#SBATCH --job-name=hys15_unit
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
export CORRECTED_STUDY_ROOT="$1"
export TMPDIR="$1/tmp/unit_${SLURM_JOB_ID}"
export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"
export JULIA_NUM_THREADS=2 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
module purge
module load community-modules julia/1.10.4
cd "$1/runtime"
julia --startup-file=no --project=. "$1/unit_tests.jl"
