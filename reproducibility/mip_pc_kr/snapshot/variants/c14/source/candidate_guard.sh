#!/bin/bash
gom_production_resolve_task() {
    test "$STUDY_ROOT" = '/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/corrected_hysteresis_15case_20260911_v1/variants/high_hybrid_wrr30_fault_nohys' || return 1
    test "${SLURM_ARRAY_TASK_ID:-}" = '848' || return 1
    test "$GOM_GRID_ROOT" = "$STUDY_ROOT" || return 1
    (cd "$STUDY_ROOT" && sha256sum --check STUDY_SHA256.txt) || return 1
    source "$STUDY_ROOT/locked.env"
    test "$GOM_PRODUCTION_CASE_KEY" = 's03_c003_case08' || return 1
    sha256sum --check "$STUDY_ROOT/INPUTS_SHA256.txt" || return 1
    test "$(git -C "$JUTULDARCY_COMBINED_REPO" rev-parse HEAD)" = "$GOM_PRODUCTION_JUTULDARCY_COMMIT" || return 1
    test -z "$(git -C "$JUTULDARCY_COMBINED_REPO" status --porcelain --untracked-files=no)" || return 1
    export TMPDIR="$STUDY_ROOT/tmp/${SLURM_JOB_ID}"
    export TMP="$TMPDIR" TEMP="$TMPDIR" TEMPDIR="$TMPDIR"
    mkdir -p "$TMPDIR"
    unset STOP_AFTER_REPORT_STEP RESTART_STEP JULIA_CMD JULIA_DEPOT_PATH CPR_PARTIAL_UPDATE GOM_PREFLIGHT_SIMULATION_REPO
    export CPR_UPDATE_INTERVAL=iteration CPR_UPDATE_INTERVAL_PARTIAL=iteration
}
