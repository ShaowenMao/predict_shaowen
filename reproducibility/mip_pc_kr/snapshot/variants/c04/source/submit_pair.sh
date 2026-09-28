#!/bin/bash
set -euo pipefail
: "${STUDY_ROOT:?}"
: "${JUTULDARCY_COMBINED_REPO:?}"
export GOM_GRID_ROOT="$STUDY_ROOT"
export GOM_PRODUCTION_MANIFEST="$STUDY_ROOT/campaign.toml"
test ! -e "$STUDY_ROOT/SUBMITTED.txt" || { echo 'Already submitted'; exit 1; }
cd "$STUDY_ROOT"
sha256sum --check STUDY_SHA256.txt
for script in source/*.sh source/*.sbatch; do bash -n "$script"; done
exec 8> .submission.lock
flock -n 8
test ! -e SUBMITTED.txt
preflight=$(sbatch --parsable --array=262 --output="$STUDY_ROOT/logs/%x-%A_%a.out" \
    --error="$STUDY_ROOT/logs/%x-%A_%a.err" --chdir="$STUDY_ROOT" \
    source/gom_step62_effective_pc_global_plateau_preflight.sbatch)
preflight=${preflight%%;*}
printf 'preflight_job=%s\n' "$preflight" > SUBMITTED.txt
export GOM_PRODUCTION_PREFLIGHT_JOB_ID="$preflight"
full=$(sbatch --parsable --array=262 --dependency="afterok:$preflight" \
    --output="$STUDY_ROOT/logs/%x-%A_%a.out" --error="$STUDY_ROOT/logs/%x-%A_%a.err" \
    --chdir="$STUDY_ROOT" source/gom_step62_effective_pc_global_plateau_full.sbatch)
full=${full%%;*}
printf 'full_job=%s\nsubmitted_utc=%s\n' "$full" "$(date -u +%FT%TZ)" >> SUBMITTED.txt
cat SUBMITTED.txt
squeue -j "$preflight,$full" -o '%.20i %.22j %.2t %.10M %.6C %R'
