#!/usr/bin/env python3
"""Merge six verified representative replay paths into a 6x87 case template."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-csv", required=True, type=Path)
    parser.add_argument("--representative-selection-csv", required=True, type=Path)
    parser.add_argument("--representative-replay-summary-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--tolerance-log10", type=float, default=1.0e-3)
    return parser.parse_args()


def read_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def main() -> int:
    args = parse_args()
    template_fields, template_rows = read_table(args.template_csv)
    _, selection_rows = read_table(args.representative_selection_csv)
    _, replay_rows = read_table(args.representative_replay_summary_csv)
    if len(template_rows) != 522:
        raise ValueError(f"Replay template has {len(template_rows)} rows, expected 522")
    if len(selection_rows) != 6 or len(replay_rows) != 6:
        raise ValueError("Representative selection and replay must each contain 6 rows")

    required_output_fields = [
        "AttemptIndex",
        "ReplayMode",
        "SmearOverlapRule",
        "CollapseAdjacentLithology",
    ]
    for field in required_output_fields:
        if field not in template_fields:
            template_fields.append(field)

    selected_case_rows: set[int] = set()
    for replay in replay_rows:
        selection_index = int(float(replay["SourceRow"]))
        if selection_index < 1 or selection_index > len(selection_rows):
            raise ValueError(f"Invalid representative SourceRow {selection_index}")
        selection = selection_rows[selection_index - 1]
        case_source_row = int(float(selection["case_source_row"]))
        if case_source_row in selected_case_rows:
            raise ValueError(f"Duplicate representative case row {case_source_row}")
        selected_case_rows.add(case_source_row)
        target = template_rows[case_source_row - 1]
        if target["TaskId"] != selection["task_id"]:
            raise ValueError(
                f"Representative task mismatch at case row {case_source_row}"
            )
        if replay["TaskId"] != selection["task_id"]:
            raise ValueError(
                f"Replay task mismatch at selection row {selection_index}"
            )
        if replay["VerificationStatus"] != "matched":
            raise ValueError(f"Representative replay did not match: {replay['TaskId']}")
        difference = float(replay["MaxAbsLog10Diff"])
        if not math.isfinite(difference) or difference > args.tolerance_log10:
            raise ValueError(
                f"Representative replay exceeds tolerance: {replay['TaskId']}"
            )
        target.update(
            {
                "VerificationStatus": replay["VerificationStatus"],
                "MaxAbsLog10Diff": replay["MaxAbsLog10Diff"],
                "OutputFile": replay["OutputFile"],
                "AttemptIndex": replay["AttemptIndex"],
                "ReplayMode": replay["ReplayMode"],
                "SmearOverlapRule": replay["SmearOverlapRule"],
                "CollapseAdjacentLithology": replay[
                    "CollapseAdjacentLithology"
                ],
            }
        )

    if len(selected_case_rows) != 6:
        raise ValueError("Did not merge exactly six representative rows")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=template_fields)
        writer.writeheader()
        writer.writerows(template_rows)
    result = {
        "status": "complete",
        "output_csv": str(args.output_csv),
        "row_count": len(template_rows),
        "representative_rows": sorted(selected_case_rows),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
