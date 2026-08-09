#!/usr/bin/env python3
"""Select Phase 2 geologies by deterministic maximin coverage.

Scientific regime classification is deliberately external to this utility.
The candidate CSV must state each geology's category and the normalized-design
features to consider. This keeps containment/migration thresholds explicit and
reviewable instead of hard-coding them in the case generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "independent_full_fault_phase2_selection_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--feature-columns", required=True)
    parser.add_argument("--categories", required=True)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--expected-total", type=int, default=12)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_selection_size(
    categories: list[str], per_category: int, expected_total: int
) -> None:
    """Enforce the frozen 12-geology Phase 2 selection design."""
    if per_category <= 0:
        raise ValueError("--per-category must be positive")
    if expected_total <= 0:
        raise ValueError("--expected-total must be positive")
    if len(set(categories)) != len(categories):
        raise ValueError("Categories must be unique")
    if len(categories) != 3 or per_category != 4:
        raise ValueError(
            "Phase 2 requires exactly three reviewed categories with "
            "four geologies per category"
        )
    actual_total = len(categories) * per_category
    if actual_total != expected_total:
        raise ValueError(
            "Phase 2 selection must match the frozen design size: "
            f"{len(categories)} categories x {per_category} per category "
            f"= {actual_total}, expected {expected_total}"
        )


def euclidean(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(first, second)))


def normalize_features(rows: list[dict[str, str]], features: list[str]) -> dict[str, list[float]]:
    values = [[float(row[column]) for column in features] for row in rows]
    if any(not math.isfinite(value) for vector in values for value in vector):
        raise ValueError("All maximin feature values must be finite")
    minimum = [min(vector[index] for vector in values) for index in range(len(features))]
    maximum = [max(vector[index] for vector in values) for index in range(len(features))]
    normalized: dict[str, list[float]] = {}
    for row, vector in zip(rows, values):
        normalized[row["geology_id"]] = [
            0.0 if maximum[index] == minimum[index] else
            (value - minimum[index]) / (maximum[index] - minimum[index])
            for index, value in enumerate(vector)
        ]
    return normalized


def select_maximin(
    rows: list[dict[str, str]],
    vectors: dict[str, list[float]],
    count: int,
) -> list[tuple[dict[str, str], float]]:
    if len(rows) < count:
        raise ValueError(f"Category has {len(rows)} candidates but requires {count}")
    ordered = sorted(rows, key=lambda row: row["geology_id"])
    centroid = [
        sum(vectors[row["geology_id"]][index] for row in ordered) / len(ordered)
        for index in range(len(next(iter(vectors.values()))))
    ]
    centroid_distances = {
        row["geology_id"]: euclidean(vectors[row["geology_id"]], centroid)
        for row in ordered
    }
    maximum_centroid_distance = max(centroid_distances.values())
    first = min(
        (row for row in ordered
         if centroid_distances[row["geology_id"]] == maximum_centroid_distance),
        key=lambda row: row["geology_id"],
    )
    selected = [(first, centroid_distances[first["geology_id"]])]
    remaining = [row for row in ordered if row is not first]
    while len(selected) < count:
        scored = []
        for row in remaining:
            distance = min(
                euclidean(vectors[row["geology_id"]], vectors[item[0]["geology_id"]])
                for item in selected
            )
            scored.append((distance, row["geology_id"], row))
        best_distance = max(item[0] for item in scored)
        best = min(
            (item for item in scored if item[0] == best_distance),
            key=lambda item: item[1],
        )
        selected.append((best[2], best[0]))
        remaining.remove(best[2])
    return selected


def main() -> int:
    args = parse_args()
    features = [item.strip() for item in args.feature_columns.split(",") if item.strip()]
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    if not features or not categories:
        raise ValueError("Feature columns and categories cannot be empty")
    validate_selection_size(categories, args.per_category, args.expected_total)
    source = args.candidates_csv.resolve()
    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        candidate_columns = list(reader.fieldnames or [])
        required = {"geology_id", "category", *features}
        missing = required - set(candidate_columns)
        if missing:
            raise ValueError(f"Candidate table lacks columns: {sorted(missing)}")
        rows = list(reader)
    if len({row["geology_id"] for row in rows}) != len(rows):
        raise ValueError("Candidate geology IDs must be unique")
    unknown = sorted({row["category"] for row in rows} - set(categories))
    if unknown:
        raise ValueError(f"Unexpected candidate categories: {unknown}")
    vectors = normalize_features(rows, features)

    selected_rows: list[dict[str, object]] = []
    for category in categories:
        candidates = [row for row in rows if row["category"] == category]
        for rank, (row, distance) in enumerate(
            select_maximin(candidates, vectors, args.per_category), start=1
        ):
            selected_rows.append(
                {
                    **row,
                    "selection_rank_within_category": rank,
                    "maximin_distance_at_selection": distance,
                }
            )

    if len(selected_rows) != args.expected_total:
        raise ValueError(
            f"Phase 2 selected {len(selected_rows)} geologies; "
            f"expected {args.expected_total}"
        )

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    temporary = output_root.with_name(f"{output_root.name}.building.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    output_path = temporary / "deep_dive_selection.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected_rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates_csv": str(source),
        "candidates_csv_sha256": sha256_file(source),
        "feature_columns": features,
        "candidate_columns_retained": candidate_columns,
        "categories": categories,
        "per_category": args.per_category,
        "expected_geology_count": args.expected_total,
        "selected_geology_count": len(selected_rows),
        "selection_method": "deterministic_category_stratified_maximin_v1",
    }
    (temporary / "deep_dive_selection_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output_root)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
