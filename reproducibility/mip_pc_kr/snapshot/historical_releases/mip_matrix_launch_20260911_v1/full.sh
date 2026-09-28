#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --account=mit_amf_advanced_cpu
#SBATCH --qos=mit_amf_advanced_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=18G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
set -euo pipefail
export STUDY_ROOT="$1" GOM_GRID_ROOT="$1" GOM_PRODUCTION_PREFLIGHT_JOB_ID="$2"
export JUTULDARCY_COMBINED_REPO=/home/shaowen/orcd/scratch/gom_grid/code/immutable/JutulDarcy_ed81af8e
test -s "$STUDY_ROOT/qualified.json"
exec bash "$STUDY_ROOT/source/gom_step62_effective_pc_global_plateau_full.sbatch"
