#!/usr/bin/env python3
"""Build small per-geology assignment files and case work manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-csv", required=True, type=Path)
    parser.add_argument("--assignment-to-task-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-filter-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_filter(path: Path | None) -> set[tuple[str, int]] | None:
    if path is None:
        return None
    result: set[tuple[str, int]] = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            result.add((row["geology_id"], int(row["case_id"])))
    if not result:
        raise ValueError("Case filter is empty")
    return result


def main() -> int:
    args = parse_args()
    sampling_path = args.sampling_csv.resolve()
    assignment_path = args.assignment_to_task_csv.resolve()
    output_root = args.output_root.resolve()
    for path in (sampling_path, assignment_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    case_filter = read_filter(args.case_filter_csv)

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    assignment_root = output_root / "geology_assignments"
    assignment_root.mkdir(parents=True)

    streams: dict[str, object] = {}
    writers: dict[str, csv.DictWriter] = {}
    geology_metadata: dict[str, dict[str, str]] = {}
    case_metadata: dict[tuple[str, int], dict[str, str]] = {}
    row_counts: dict[str, int] = {}
    combined_fields: list[str] | None = None

    try:
        with sampling_path.open(
            newline="", encoding="utf-8-sig"
        ) as sampling_stream, assignment_path.open(
            newline="", encoding="utf-8-sig"
        ) as assignment_stream:
            sampling_reader = csv.DictReader(sampling_stream)
            assignment_reader = csv.DictReader(assignment_stream)
            combined_fields = list(sampling_reader.fieldnames or []) + [
                "assignment_index",
                "task_id",
                "task_key_sha256",
            ]
            for row_index, (sampling, assignment) in enumerate(
                zip(sampling_reader, assignment_reader, strict=True), start=1
            ):
                if int(assignment["assignment_index"]) != row_index:
                    raise ValueError(
                        f"Assignment index mismatch at row {row_index}"
                    )
                if sampling["geology_id"] != assignment["geology_id"]:
                    raise ValueError(f"Geology mismatch at row {row_index}")
                if sampling["case_id"] != assignment["case_id"]:
                    raise ValueError(f"Case mismatch at row {row_index}")
                if sampling["window"] != assignment["window"]:
                    raise ValueError(f"Window mismatch at row {row_index}")

                geology_id = sampling["geology_id"]
                case_id = int(sampling["case_id"])
                if case_filter is not None and (geology_id, case_id) not in case_filter:
                    continue
                if geology_id not in writers:
                    path = assignment_root / f"{geology_id}.csv"
                    stream = path.open("w", newline="", encoding="utf-8")
                    writer = csv.DictWriter(stream, fieldnames=combined_fields)
                    writer.writeheader()
                    streams[geology_id] = stream
                    writers[geology_id] = writer
                    row_counts[geology_id] = 0
                combined = dict(sampling)
                combined.update(
                    {
                        "assignment_index": assignment["assignment_index"],
                        "task_id": assignment["task_id"],
                        "task_key_sha256": assignment["task_key_sha256"],
                    }
                )
                writers[geology_id].writerow(combined)
                row_counts[geology_id] += 1
                geology_metadata.setdefault(geology_id, combined)
                case_metadata.setdefault((geology_id, case_id), combined)
    finally:
        for stream in streams.values():
            stream.close()

    expected_case_rows = 87 * 6
    expected_geology_rows = expected_case_rows * 10
    for geology_id, count in row_counts.items():
        expected = (
            expected_geology_rows
            if case_filter is None
            else expected_case_rows
            * sum(1 for key in case_metadata if key[0] == geology_id)
        )
        if count != expected:
            raise ValueError(
                f"{geology_id} has {count} assignment rows; expected {expected}"
            )

    geology_rows = []
    for index, geology_id in enumerate(sorted(geology_metadata), start=1):
        row = geology_metadata[geology_id]
        path = assignment_root / f"{geology_id}.csv"
        case_ids = sorted(case_id for geo, case_id in case_metadata if geo == geology_id)
        geology_rows.append(
            {
                "geology_work_index": index,
                "geology_id": geology_id,
                "scenario_index": row["scenario_index"],
                "scenario_label": row["scenario_label"],
                "case_ids": ",".join(map(str, case_ids)),
                "assignment_count": row_counts[geology_id],
                "assignment_relative_path": path.relative_to(output_root).as_posix(),
                "assignment_sha256": sha256_file(path),
            }
        )

    case_rows = []
    for index, key in enumerate(sorted(case_metadata), start=1):
        geology_id, case_id = key
        row = case_metadata[key]
        case_rows.append(
            {
                "case_work_index": index,
                "geology_id": geology_id,
                "scenario_index": row["scenario_index"],
                "scenario_label": row["scenario_label"],
                "case_id": case_id,
                "case_name": row["case_name"],
                "case_category": row["case_category"],
                "case_relative_path": f"cases/{geology_id}/case{case_id:02d}",
            }
        )

    with (output_root / "geology_work.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geology_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geology_rows)
    with (output_root / "case_work.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(case_rows[0].keys()))
        writer.writeheader()
        writer.writerows(case_rows)

    metadata = {
        "geology_count": len(geology_rows),
        "case_count": len(case_rows),
        "assignment_count": sum(row_counts.values()),
        "case_filter_csv": (
            str(args.case_filter_csv.resolve()) if args.case_filter_csv else ""
        ),
    }
    (output_root / "case_work_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
