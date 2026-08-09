#!/usr/bin/env python3
"""Analyze balanced Phase 1 reservoir results without mixing case roles.

The input result table is long-form with one row per geology, sampling case,
and quantity of interest (QoI). Case roles are read from the canonical
sampling manifest rather than trusted from the result table. Only the twelve
``independent_full`` cases contribute to probabilistic moments and the
design-based Level-1/Level-2 decomposition. Deterministic medoid and stress
cases are reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, variance


SCHEMA_VERSION = "independent_full_fault_phase1_analysis_v1"
RESULT_COLUMNS = {"geology_id", "case_id", "qoi_name", "qoi_value"}
MANIFEST_COLUMNS = {
    "design_version",
    "geology_id",
    "phase",
    "case_id",
    "case_type",
    "replicate_id",
    "window_id",
    "slice_id",
    "use_for_probabilistic_uq",
    "is_benchmark",
    "is_stress_test",
}


@dataclass(frozen=True)
class CaseRole:
    geology_id: str
    case_id: str
    case_type: str
    replicate_id: int
    use_for_probabilistic_uq: bool
    is_benchmark: bool
    is_stress_test: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--sampling-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--expected-geologies", type=int, default=162)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid Boolean value: {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_roles(path: Path) -> dict[tuple[str, str], CaseRole]:
    roles: dict[tuple[str, str], CaseRole] = {}
    coverage: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sampling manifest lacks columns: {sorted(missing)}")
        for row in reader:
            if row["design_version"] != "independent_full_fault_v1":
                raise ValueError(f"Unexpected design version: {row['design_version']}")
            if row["phase"] != "phase1":
                continue
            key = (row["geology_id"], row["case_id"])
            role = CaseRole(
                geology_id=row["geology_id"],
                case_id=row["case_id"],
                case_type=row["case_type"],
                replicate_id=int(row["replicate_id"]),
                use_for_probabilistic_uq=parse_bool(row["use_for_probabilistic_uq"]),
                is_benchmark=parse_bool(row["is_benchmark"]),
                is_stress_test=parse_bool(row["is_stress_test"]),
            )
            if key in roles and roles[key] != role:
                raise ValueError(f"Inconsistent role metadata for {key}")
            roles[key] = role
            coverage[key].add((row["window_id"], int(row["slice_id"])))

    expected_coordinates = {
        (f"famp{window}", slice_id)
        for window in range(1, 7)
        for slice_id in range(1, 88)
    }
    for key, coordinates in coverage.items():
        if coordinates != expected_coordinates:
            raise ValueError(f"Sampling case lacks exact 6 x 87 coverage: {key}")
    if not roles:
        raise ValueError("No Phase 1 case roles found")
    return roles


def validate_geology_design(roles: dict[tuple[str, str], CaseRole]) -> None:
    by_geology: dict[str, list[CaseRole]] = defaultdict(list)
    for role in roles.values():
        by_geology[role.geology_id].append(role)
    for geology_id, geology_roles in by_geology.items():
        independent = sorted(
            role.replicate_id
            for role in geology_roles
            if role.case_type == "independent_full"
        )
        deterministic = sorted(
            role.case_type
            for role in geology_roles
            if role.case_type != "independent_full"
        )
        if independent != list(range(1, 13)):
            raise ValueError(f"{geology_id} does not have independent replicates 1:12")
        if deterministic != [
            "high_state_stress",
            "low_state_stress",
            "representative_medoid",
        ]:
            raise ValueError(f"{geology_id} has invalid deterministic cases")
        for role in geology_roles:
            expected = {
                "independent_full": (True, False, False),
                "representative_medoid": (False, True, False),
                "low_state_stress": (False, False, True),
                "high_state_stress": (False, False, True),
            }[role.case_type]
            actual = (
                role.use_for_probabilistic_uq,
                role.is_benchmark,
                role.is_stress_test,
            )
            if actual != expected:
                raise ValueError(f"Incorrect case-role flags for {role.case_id}")


def validate_analysis_rectangle(
    roles: dict[tuple[str, str], CaseRole],
    summaries: list[dict[str, object]],
    expected_geologies: int,
) -> None:
    """Require every QoI to cover the complete balanced Phase 1 design."""
    if expected_geologies <= 0:
        raise ValueError("--expected-geologies must be positive")
    role_geologies = {role.geology_id for role in roles.values()}
    if len(role_geologies) != expected_geologies:
        raise ValueError(
            "Phase 1 manifest geology count mismatch: "
            f"expected {expected_geologies}, found {len(role_geologies)}"
        )
    summary_geologies = {str(row["geology_id"]) for row in summaries}
    if summary_geologies != role_geologies:
        missing = sorted(role_geologies - summary_geologies)
        extra = sorted(summary_geologies - role_geologies)
        raise ValueError(
            "Phase 1 result geology rectangle is incomplete; "
            f"missing={missing}, extra={extra}"
        )
    by_qoi: dict[str, set[str]] = defaultdict(set)
    for row in summaries:
        by_qoi[str(row["qoi_name"])].add(str(row["geology_id"]))
    for qoi_name, geologies in sorted(by_qoi.items()):
        if geologies != role_geologies:
            missing = sorted(role_geologies - geologies)
            raise ValueError(
                f"QoI {qoi_name!r} lacks the full Phase 1 geology rectangle; "
                f"missing={missing}"
            )


def read_results(
    path: Path, roles: dict[tuple[str, str], CaseRole]
) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = RESULT_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Result table lacks columns: {sorted(missing)}")
        for row in reader:
            case_key = (row["geology_id"], row["case_id"])
            if case_key not in roles:
                raise ValueError(f"Result references unknown Phase 1 case: {case_key}")
            qoi_name = row["qoi_name"].strip()
            if not qoi_name:
                raise ValueError("qoi_name cannot be empty")
            value = float(row["qoi_value"])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite result for {case_key}, {qoi_name}")
            analysis_key = (row["geology_id"], qoi_name)
            if row["case_id"] in grouped[analysis_key]:
                raise ValueError(f"Duplicate result for {case_key}, {qoi_name}")
            grouped[analysis_key][row["case_id"]] = value
    if not grouped:
        raise ValueError("Result table is empty")
    return grouped


def midrank_percentile(sample: list[float], value: float) -> float:
    less = sum(item < value for item in sample)
    equal = sum(item == value for item in sample)
    return 100.0 * (less + 0.5 * equal) / len(sample)


def summarize_geologies(
    roles: dict[tuple[str, str], CaseRole],
    results: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, object]]:
    role_by_geology: dict[str, list[CaseRole]] = defaultdict(list)
    for role in roles.values():
        role_by_geology[role.geology_id].append(role)
    rows: list[dict[str, object]] = []
    for (geology_id, qoi_name), case_values in sorted(results.items()):
        geology_roles = role_by_geology[geology_id]
        expected_ids = {role.case_id for role in geology_roles}
        if set(case_values) != expected_ids:
            missing = sorted(expected_ids - set(case_values))
            extra = sorted(set(case_values) - expected_ids)
            raise ValueError(
                f"Incomplete results for {geology_id}, {qoi_name}; "
                f"missing={missing}, extra={extra}"
            )
        independent_roles = sorted(
            (role for role in geology_roles if role.use_for_probabilistic_uq),
            key=lambda role: role.replicate_id,
        )
        independent = [case_values[role.case_id] for role in independent_roles]
        by_type = {role.case_type: case_values[role.case_id] for role in geology_roles}
        medoid = by_type["representative_medoid"]
        low = by_type["low_state_stress"]
        high = by_type["high_state_stress"]
        rows.append(
            {
                "geology_id": geology_id,
                "qoi_name": qoi_name,
                "n_independent": len(independent),
                "independent_mean": mean(independent),
                "independent_sample_variance": variance(independent),
                "independent_sample_std": math.sqrt(variance(independent)),
                "independent_min": min(independent),
                "independent_max": max(independent),
                "representative_medoid_value": medoid,
                "representative_medoid_bias": medoid - mean(independent),
                "representative_medoid_midrank_percentile": midrank_percentile(
                    independent, medoid
                ),
                "low_state_stress_value": low,
                "high_state_stress_value": high,
                "stress_delta_high_minus_low": high - low,
            }
        )
    return rows


def variance_decomposition(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_qoi: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        by_qoi[str(row["qoi_name"])].append(row)
    output: list[dict[str, object]] = []
    for qoi_name, rows in sorted(by_qoi.items()):
        if len(rows) < 2:
            raise ValueError(
                f"At least two geologies are required for decomposition of {qoi_name}"
            )
        counts = {int(row["n_independent"]) for row in rows}
        if counts != {12}:
            raise ValueError(f"Balanced decomposition requires n=12 for {qoi_name}")
        within = mean(float(row["independent_sample_variance"]) for row in rows)
        between_raw = variance(float(row["independent_mean"]) for row in rows)
        finite_noise = mean(
            float(row["independent_sample_variance"]) / 12.0 for row in rows
        )
        level1 = max(0.0, between_raw - finite_noise)
        total = level1 + within
        output.append(
            {
                "qoi_name": qoi_name,
                "n_geologies": len(rows),
                "n_independent_per_geology": 12,
                "level2_within_geology_variance": within,
                "raw_variance_of_geology_means": between_raw,
                "finite_replicate_noise_correction": finite_noise,
                "level1_geologic_design_variance": level1,
                "level1_design_contribution": level1 / total if total > 0 else 0.0,
                "level2_design_contribution": within / total if total > 0 else 0.0,
                "interpretation": (
                    "uniform-design contributions over the prescribed geologic scenarios"
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    results_path = args.results_csv.resolve()
    manifest_path = args.sampling_manifest.resolve()
    for path in (results_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    temporary = output_root.with_name(f"{output_root.name}.building.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    roles = read_case_roles(manifest_path)
    validate_geology_design(roles)
    results = read_results(results_path, roles)
    summaries = summarize_geologies(roles, results)
    validate_analysis_rectangle(roles, summaries, args.expected_geologies)
    decomposition = variance_decomposition(summaries)
    summary_path = temporary / "phase1_geology_qoi_summary.csv"
    decomposition_path = temporary / "phase1_variance_decomposition.csv"
    write_csv(summary_path, summaries)
    write_csv(decomposition_path, decomposition)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": sha256_file(manifest_path),
        "results_csv": str(results_path),
        "results_csv_sha256": sha256_file(results_path),
        "geology_count": len({row["geology_id"] for row in summaries}),
        "expected_geology_count": args.expected_geologies,
        "qoi_count": len({row["qoi_name"] for row in summaries}),
        "probabilistic_case_type": "independent_full",
        "excluded_from_probabilistic_moments": [
            "representative_medoid",
            "low_state_stress",
            "high_state_stress",
        ],
    }
    (temporary / "phase1_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output_root)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
