#!/usr/bin/env bash
# Summarize Engaging case01 replay/Pc/Kr outputs and runtime diagnostics.

set -euo pipefail

RUN_ID="${1:-${RUN_ID:-}}"
if [[ -z "${RUN_ID}" ]]; then
    echo "Usage: $0 RUN_ID" >&2
    exit 2
fi

SCRATCH_ROOT="${SCRATCH_ROOT:-/home/shaowen/orcd/scratch/predict_shaowen}"
RUN_ROOT="${RUN_ROOT:-${SCRATCH_ROOT}/runs/${RUN_ID}}"

echo "RUN_ID=${RUN_ID}"
echo "RUN_ROOT=${RUN_ROOT}"
echo

echo "=== Slurm jobs ==="
sacct -j "$(find "${RUN_ROOT}/slurm" -maxdepth 1 -type f -name '*.out' -printf '%f\n' \
    | sed -E 's/.*_([0-9]+)\.out/\1/' | sort -u | paste -sd, -)" \
    --format=JobID,JobName%30,State,ExitCode,Elapsed,AllocCPUS,MaxRSS -P 2>/dev/null || true
echo

echo "=== Stage status ==="
find "${RUN_ROOT}/status" -maxdepth 1 -type f -printf '%f\n' 2>/dev/null | sort || true
echo

echo "=== Stage diagnostics ==="
find "${RUN_ROOT}/diagnostics" -maxdepth 1 -type f -name '*.status' -print 2>/dev/null \
    | sort | while read -r file; do
        echo "--- $(basename "${file}") ---"
        grep -E 'stage=|job_id=|exit_code=|elapsed_seconds=|finished_at=' "${file}" || true
    done
echo

echo "=== Output row counts ==="
find "${RUN_ROOT}" -type f \( \
        -name 'replay_summary_with_full87_context_*.csv' -o \
        -name 'pc_curve_summary_*.csv' -o \
        -name 'pc_curve_points_*.csv' -o \
        -name 'kr_curve_summary_*.csv' -o \
        -name 'kr_curve_points_*.csv' \) \
    | sort | while read -r file; do
        printf '%7d  %s\n' "$(wc -l < "${file}")" "${file#${RUN_ROOT}/}"
    done
echo

echo "=== Kr timing summary ==="
python3 - "$RUN_ROOT" <<'PY'
import csv
import glob
import math
import os
import statistics
import sys

run_root = sys.argv[1]
paths = glob.glob(os.path.join(run_root, "kr_dyn*", "tables", "kr_curve_summary_*.csv"))
if not paths:
    print("No Kr summary table found yet.")
    raise SystemExit

for path in sorted(paths):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"--- {os.path.relpath(path, run_root)} ---")
    print(f"curves={len(rows)}")
    for col in [
        "TotalCurveSeconds",
        "PcUpscalingSeconds",
        "DynamicKrTotalSeconds",
        "Dynamic3DSeconds",
        "Dynamic1DMatchSeconds",
        "CheckpointSaveSeconds",
    ]:
        vals = []
        for row in rows:
            try:
                val = float(row.get(col, "nan"))
            except ValueError:
                val = math.nan
            if math.isfinite(val):
                vals.append(val)
        if not vals:
            continue
        vals_sorted = sorted(vals)
        q50 = statistics.median(vals_sorted)
        q90 = vals_sorted[min(len(vals_sorted) - 1, int(math.ceil(0.90 * len(vals_sorted)) - 1))]
        q95 = vals_sorted[min(len(vals_sorted) - 1, int(math.ceil(0.95 * len(vals_sorted)) - 1))]
        print(
            f"{col}: n={len(vals)} min={min(vals):.3g} "
            f"median={q50:.3g} p90={q90:.3g} p95={q95:.3g} max={max(vals):.3g}"
        )

    def total_seconds(row):
        try:
            return float(row.get("TotalCurveSeconds", "nan"))
        except ValueError:
            return math.nan

    slow = sorted(
        [r for r in rows if math.isfinite(total_seconds(r))],
        key=total_seconds,
        reverse=True,
    )[:10]
    if slow:
        print("Slowest curves:")
        for row in slow:
            print(
                "  curve={ProductionCurveId} case={Level3CaseId} "
                "slice={SliceIndex} window={Window} total={TotalCurveSeconds}s "
                "dyn3d={Dynamic3DSeconds}s dyn1d={Dynamic1DMatchSeconds}s".format(**row)
            )
PY
