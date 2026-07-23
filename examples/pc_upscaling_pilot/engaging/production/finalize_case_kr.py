#!/usr/bin/env python3
"""Validate and atomically publish one production dynamic-Kr case."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geology-id", required=True)
    parser.add_argument("--case-id", required=True, type=int)
    parser.add_argument("--kr-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--physics-commit", required=True)
    parser.add_argument("--method-config-sha256", required=True)
    parser.add_argument("--overwrite-incomplete", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def find_single(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one file matching {root / pattern}, found {len(matches)}"
        )
    return matches[0]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def finite(value: str, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {value!r}")
    return result


def validate_representative_curves(
    summary_path: Path,
    points_path: Path,
    geology_id: str,
    case_id: int,
) -> dict:
    summary = read_rows(summary_path)
    if len(summary) != 6:
        raise ValueError(f"Dynamic Kr summary has {len(summary)} rows, expected 6")
    windows = {row["Window"] for row in summary}
    if windows != {f"famp{i}" for i in range(1, 7)}:
        raise ValueError(f"Dynamic Kr window coverage is invalid: {sorted(windows)}")
    for row in summary:
        if row["GeologyId"] != geology_id:
            raise ValueError("Dynamic Kr geology does not match requested case")
        if int(float(row["Level3CaseId"])) != case_id:
            raise ValueError("Dynamic Kr case ID does not match requested case")
        if int(float(row["OriginalCartDimY"])) != int(float(row["RuntimeCartDimY"])):
            raise ValueError("Strike dimension changed during dynamic Kr upscaling")
        if int(float(row["RuntimeCartDimY"])) <= 1:
            raise ValueError("Dynamic Kr used a strike-collapsed grid")
        swi = finite(row["IrreducibleWaterSaturation"], "Swi")
        sg_max = finite(row["PcMaxSg"], "PcMaxSg")
        if not 0.0 <= swi <= 1.0 or not 0.0 <= sg_max <= 1.0:
            raise ValueError("Dynamic Kr saturation endpoint is outside [0,1]")
        if abs(swi - (1.0 - sg_max)) > 1.0e-8:
            raise ValueError("Dynamic Kr Pc/Swi endpoint mismatch")
        if finite(row["HistoryMatchError"], "HistoryMatchError") < 0.0:
            raise ValueError("Dynamic Kr history-match error is negative")

    points = read_rows(points_path)
    by_curve: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for row in points:
        curve_id = int(float(row["ProductionCurveId"]))
        sg = finite(row["Sg"], "Sg")
        krg = finite(row["Krg"], "Krg")
        krw = finite(row["Krw"], "Krw")
        if not 0.0 <= sg <= 1.0:
            raise ValueError("Dynamic Kr gas saturation is outside [0,1]")
        if not -1.0e-10 <= krg <= 1.0 + 1.0e-10:
            raise ValueError("Krg is outside physical [0,1] bounds")
        if not -1.0e-10 <= krw <= 1.0 + 1.0e-10:
            raise ValueError("Krw is outside physical [0,1] bounds")
        by_curve[curve_id].append((sg, krg, krw))
    if len(by_curve) != 6:
        raise ValueError(f"Expected 6 representative Kr curves, found {len(by_curve)}")
    for curve_id, values in by_curve.items():
        values.sort()
        for previous, current in zip(values, values[1:]):
            if current[1] + 1.0e-8 < previous[1]:
                raise ValueError(f"Krg is non-monotone for curve {curve_id}")
            if current[2] - 1.0e-8 > previous[2]:
                raise ValueError(f"Krw is non-monotone for curve {curve_id}")
    return {
        "summary_rows": len(summary),
        "curve_point_rows": len(points),
        "windows": sorted(windows),
    }


def validate_slice_curves(path: Path) -> dict:
    rows = read_rows(path)
    expected = 522 * 101
    if len(rows) != expected:
        raise ValueError(f"Slice Kr table has {len(rows)} rows, expected {expected}")
    keys: set[tuple[int, str]] = set()
    endpoints: dict[tuple[int, str], tuple[float, float]] = {}
    for row in rows:
        key = (int(float(row["SliceIndex"])), row["Window"])
        keys.add(key)
        sg = finite(row["GasSaturation"], "GasSaturation")
        sw = finite(row["WaterSaturation"], "WaterSaturation")
        krg = finite(row["Krg"], "Krg")
        krw = finite(row["Krw"], "Krw")
        if abs(sw - (1.0 - sg)) > 1.0e-8:
            raise ValueError("Slice Kr saturation coordinates are inconsistent")
        if not 0.0 <= krg <= 1.0 or not 0.0 <= krw <= 1.0:
            raise ValueError("Slice Kr values are outside [0,1]")
        if row["IsEndpoint"].strip().lower() in {"1", "true"}:
            endpoints[key] = (sg, finite(row["EffectiveSwi"], "EffectiveSwi"))
    expected_keys = {
        (slice_id, f"famp{window_id}")
        for slice_id in range(1, 88)
        for window_id in range(1, 7)
    }
    if keys != expected_keys or set(endpoints) != expected_keys:
        raise ValueError("Slice Kr table does not cover exactly 6x87 endpoints")
    for sg, swi in endpoints.values():
        if abs(swi - (1.0 - sg)) > 1.0e-8:
            raise ValueError("Slice Kr endpoint does not match effective Swi")
    return {"row_count": len(rows), "endpoint_count": len(endpoints)}


def main() -> int:
    args = parse_args()
    kr_root = args.kr_root.resolve()
    output_dir = args.output_dir.resolve()
    done_path = output_dir / "case.done.json"
    if done_path.is_file():
        marker = json.loads(done_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "geology_id": args.geology_id,
            "case_id": args.case_id,
            "physics_commit": args.physics_commit,
            "method_config_sha256": args.method_config_sha256,
        }
        if all(marker.get(key) == value for key, value in expected.items()):
            print(f"Case already complete: {args.geology_id} case {args.case_id:02d}")
            return 0
        raise ValueError(f"Existing done marker does not match: {done_path}")
    if output_dir.exists():
        if not args.overwrite_incomplete:
            raise FileExistsError(
                f"Incomplete output exists; pass --overwrite-incomplete: {output_dir}"
            )
        shutil.rmtree(output_dir)

    summary_path = find_single(
        kr_root / "tables", "kr_curve_summary_*_dyn_swi_medoid.csv"
    )
    points_path = find_single(
        kr_root / "curves", "kr_curve_points_*_dyn_swi_medoid.csv"
    )
    slice_path = find_single(
        kr_root / "curves", "kr_slice_curves_swi_medoid_*.csv"
    )
    selection_path = find_single(
        kr_root / "tables", "kr_representative_selection_*_swi_medoid.csv"
    )
    full_mat = find_single(kr_root / "reservoir_ready", "*.mat")
    reduced_mat = find_single(
        kr_root / "reservoir_ready_pe_branch_medoid", "*.mat"
    )
    if full_mat.stat().st_size < 1024 or reduced_mat.stat().st_size < 1024:
        raise ValueError("Reservoir-ready MAT output is unexpectedly small")
    selection = read_rows(selection_path)
    if len(selection) != 6:
        raise ValueError("Dynamic Kr representative selection does not contain 6 rows")

    kr_report = validate_representative_curves(
        summary_path, points_path, args.geology_id, args.case_id
    )
    slice_report = validate_slice_curves(slice_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{args.geology_id}_case{args.case_id:02d}.partial.",
            dir=output_dir.parent,
        )
    )
    try:
        shutil.copytree(kr_root, partial_dir / "kr", dirs_exist_ok=True)
        files = {}
        for path in sorted((partial_dir / "kr").rglob("*")):
            if path.is_file():
                files[path.relative_to(partial_dir).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        marker = {
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "geology_id": args.geology_id,
            "case_id": args.case_id,
            "physics_commit": args.physics_commit,
            "method_config_sha256": args.method_config_sha256,
            "dynamic_kr_representative_count": 6,
            "pc_assignment_count": 522,
            "strike_collapse_used": False,
            "amgcl_required": True,
            "pc_representations": ["full_slice", "pe_branch_medoid"],
            "kr_validation": kr_report,
            "slice_validation": slice_report,
            "files": files,
        }
        (partial_dir / "case.done.json").write_text(
            json.dumps(marker, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(partial_dir, output_dir)
    except Exception:
        shutil.rmtree(partial_dir, ignore_errors=True)
        raise
    print(json.dumps(marker, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
