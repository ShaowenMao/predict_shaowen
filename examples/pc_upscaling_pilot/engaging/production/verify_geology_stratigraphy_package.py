#!/usr/bin/env python3
"""Validate a completed production geology-stratigraphy companion package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--expected-geologies", type=int, default=162)
    parser.add_argument("--expected-cases", type=int, default=1620)
    parser.add_argument("--expected-cases-per-geology", type=int, default=10)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def package_path(package_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else package_root / path


def validate_checksum_file(package_root: Path, checksum_file: Path) -> int:
    count = 0
    for line_number, raw_line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, relative_path = raw_line.split("  ", maxsplit=1)
        except ValueError as error:
            raise ValueError(
                f"{checksum_file}:{line_number}: invalid checksum line"
            ) from error
        path = package_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing checksummed file: {path}")
        require_hash(path, expected, str(path))
        count += 1
    if count == 0:
        raise ValueError("The checksum inventory is empty")
    return count


def main() -> int:
    args = parse_args()
    package_root = args.package_root.resolve()
    completion_path = package_root / "geology_stratigraphy.done.json"
    manifest_path = package_root / "geology_stratigraphy_manifest.csv"
    links_path = package_root / "geology_fault_case_links.csv"
    checksums_path = package_root / "SHA256SUMS"
    for path in (completion_path, manifest_path, links_path, checksums_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing package file: {path}")

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_completion = {
        "status": "complete",
        "content_type": "production_geology_stratigraphy_package",
        "expected_geology_count": args.expected_geologies,
        "generated_geology_count": args.expected_geologies,
        "expected_fault_case_count": args.expected_cases,
        "linked_fault_case_count": args.expected_cases,
        "cases_per_geology": args.expected_cases_per_geology,
        "pc_representation": "full_slice",
    }
    for field, expected in expected_completion.items():
        actual = completion.get(field)
        if actual != expected:
            raise ValueError(f"{field}={actual!r}; expected {expected!r}")

    require_hash(
        manifest_path,
        completion["geology_manifest_sha256"],
        "Geology manifest",
    )
    require_hash(
        links_path,
        completion["fault_case_links_sha256"],
        "Fault-case links",
    )
    require_hash(
        checksums_path,
        completion["checksums_sha256"],
        "Checksum inventory",
    )

    manifest = read_rows(manifest_path)
    links = read_rows(links_path)
    if len(manifest) != args.expected_geologies:
        raise ValueError(
            f"Geology manifest has {len(manifest)} rows; "
            f"expected {args.expected_geologies}"
        )
    if len(links) != args.expected_cases:
        raise ValueError(
            f"Fault-case link manifest has {len(links)} rows; "
            f"expected {args.expected_cases}"
        )

    manifest_by_geology: dict[str, dict[str, str]] = {}
    geology_hashes: set[str] = set()
    for row in manifest:
        geology_id = row["GeologyId"]
        if geology_id in manifest_by_geology:
            raise ValueError(f"Duplicate geology manifest row: {geology_id}")
        geology_hash = row["GeologyHash"]
        if len(geology_hash) != 64:
            raise ValueError(f"Invalid geology hash for {geology_id}")
        if geology_hash in geology_hashes:
            raise ValueError(f"Duplicate geology hash for {geology_id}")
        geology_hashes.add(geology_hash)
        manifest_by_geology[geology_id] = row

        mat_path = package_path(package_root, row["StratigraphyMat"])
        summary_path = package_path(package_root, row["LayerSummaryCsv"])
        link_path = package_path(package_root, row["FaultCaseLinkCsv"])
        for path in (mat_path, summary_path, link_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing geology artifact: {path}")
        if mat_path.stat().st_size != int(row["StratigraphyMatBytes"]):
            raise ValueError(f"Stratigraphy MAT byte mismatch: {mat_path}")
        require_hash(
            mat_path,
            row["StratigraphyMatSha256"],
            f"Stratigraphy MAT {geology_id}",
        )
        if int(row["LinkedFaultCaseCount"]) != args.expected_cases_per_geology:
            raise ValueError(f"Wrong linked case count for {geology_id}")

        local_links = read_rows(link_path)
        if len(local_links) != args.expected_cases_per_geology:
            raise ValueError(f"Wrong per-geology link count for {geology_id}")
        if {item["GeologyId"] for item in local_links} != {geology_id}:
            raise ValueError(f"Per-geology link identity mismatch: {geology_id}")
        if {item["GeologyHash"] for item in local_links} != {geology_hash}:
            raise ValueError(f"Per-geology hash linkage mismatch: {geology_id}")

    case_ids_by_geology: dict[str, set[int]] = defaultdict(set)
    key_counts: Counter[tuple[str, int]] = Counter()
    for row in links:
        geology_id = row["GeologyId"]
        if geology_id not in manifest_by_geology:
            raise ValueError(f"Unknown linked geology: {geology_id}")
        if row["GeologyHash"] != manifest_by_geology[geology_id]["GeologyHash"]:
            raise ValueError(f"Linked geology hash mismatch: {geology_id}")
        case_id = int(float(row["Level3CaseId"]))
        case_ids_by_geology[geology_id].add(case_id)
        key_counts[(geology_id, case_id)] += 1
        if row["PcRepresentation"] != "full_slice":
            raise ValueError(
                f"{geology_id} case {case_id}: expected full_slice representation"
            )
        if row["GeologyIdVerified"].lower() not in {"1", "true"}:
            raise ValueError(f"{geology_id} case {case_id}: geology not verified")
        if row["ReadableVerified"].lower() not in {"1", "true"}:
            raise ValueError(f"{geology_id} case {case_id}: MAT not verified")
        if row["CaseCompletionGateValidated"].lower() not in {"1", "true"}:
            raise ValueError(f"{geology_id} case {case_id}: gate not verified")
        fault_path = Path(row["FaultInputFolder"]) / row["FaultInputFile"]
        if not fault_path.is_file():
            raise FileNotFoundError(f"Missing linked fault MAT: {fault_path}")
        if fault_path.stat().st_size != int(row["FileSizeBytes"]):
            raise ValueError(f"Linked fault MAT byte mismatch: {fault_path}")

    expected_case_ids = set(range(1, args.expected_cases_per_geology + 1))
    for geology_id in manifest_by_geology:
        if case_ids_by_geology[geology_id] != expected_case_ids:
            raise ValueError(f"Case coverage mismatch for {geology_id}")
    duplicates = [key for key, count in key_counts.items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate or missing case links: {duplicates[:5]}")

    checksum_count = validate_checksum_file(package_root, checksums_path)
    report = {
        "status": "complete",
        "package_root": str(package_root),
        "geology_count": len(manifest),
        "fault_case_link_count": len(links),
        "cases_per_geology": args.expected_cases_per_geology,
        "checksummed_package_file_count": checksum_count,
        "full_slice_only": True,
        "fault_property_files_modified": False,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
