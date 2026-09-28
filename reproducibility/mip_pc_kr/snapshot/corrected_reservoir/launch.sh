#!/bin/bash
set -euo pipefail
export STUDY_ROOT="$1" GOM_GRID_ROOT="$1"
export CORRECTED_STUDY_ROOT="$(dirname "$(dirname "$1")")"
export JUTULDARCY_COMBINED_REPO="$CORRECTED_STUDY_ROOT/runtime"
unset JULIA_DEPOT_PATH GOM_PREFLIGHT_SIMULATION_REPO STOP_AFTER_REPORT_STEP RESTART_STEP
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case "$2" in
    preflight) exec bash "$STUDY_ROOT/source/gom_step62_effective_pc_global_plateau_preflight.sbatch" ;;
    full)
        export GOM_PRODUCTION_PREFLIGHT_JOB_ID="$3"
        exec bash "$STUDY_ROOT/source/gom_step62_effective_pc_global_plateau_full.sbatch"
        ;;
    *) exit 2 ;;
esac
