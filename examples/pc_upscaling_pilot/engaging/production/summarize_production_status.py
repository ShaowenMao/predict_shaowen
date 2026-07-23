#!/usr/bin/env python3
"""Summarize restartable checkpoint and case completion for one production run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def read_csv_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def marker_status(paths: list[Path], expected_identity: str) -> Counter:
    counts: Counter = Counter()
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("status") != "complete":
                counts["invalid_status"] += 1
            elif expected_identity not in data:
                counts["missing_identity"] += 1
            else:
                counts["complete"] += 1
        except Exception:
            counts["unreadable"] += 1
    return counts


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    checkpoint_expected = read_csv_count(
        run_root / "checkpoint_manifest" / "checkpoint_groups.csv"
    )
    case_expected = read_csv_count(
        run_root / "case_work_manifest" / "case_work.csv"
    )
    geology_expected = read_csv_count(
        run_root / "case_work_manifest" / "geology_work.csv"
    )

    checkpoint_markers = list(
        (run_root / "checkpoint_pc").glob("checkpoint_*/checkpoint.done.json")
    )
    case_input_markers = list(
        (run_root / "case_inputs" / "cases").glob(
            "*/case*/case_inputs.done.json"
        )
    )
    case_result_markers = list(
        (run_root / "case_results" / "cases").glob("*/case*/case.done.json")
    )
    report = {
        "run_root": str(run_root),
        "checkpoint": {
            "expected": checkpoint_expected,
            "markers_found": len(checkpoint_markers),
            "status": marker_status(checkpoint_markers, "group_id"),
        },
        "case_inputs": {
            "expected_geologies": geology_expected,
            "expected_cases": case_expected,
            "markers_found": len(case_input_markers),
            "status": marker_status(case_input_markers, "case_id"),
        },
        "case_results": {
            "expected": case_expected,
            "markers_found": len(case_result_markers),
            "status": marker_status(case_result_markers, "case_id"),
        },
    }
    output_path = args.output_json or run_root / "production_status.json"
    output_path.write_text(
        json.dumps(report, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=dict))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
