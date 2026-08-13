"""Regression tests for noncontiguous stratigraphy-package case IDs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("verify_geology_stratigraphy_package.py")
CASE_IDS = (1, 101, 103)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class StratigraphyPackageVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        geology_id = "s01_c001"
        geology_hash = "a" * 64
        geology_root = self.root / "geologies" / geology_id
        geology_root.mkdir(parents=True)
        stratigraphy = geology_root / f"geology_stratigraphy_{geology_id}.mat"
        summary = geology_root / f"geology_stratigraphy_summary_{geology_id}.csv"
        local_links = geology_root / f"geology_fault_case_links_{geology_id}.csv"
        stratigraphy.write_bytes(b"stratigraphy")
        summary.write_text("window,lithology\nW1,sand\n", encoding="utf-8")

        links: list[dict[str, object]] = []
        for case_id in CASE_IDS:
            fault_root = self.root / "faults" / f"case{case_id:03d}"
            fault_root.mkdir(parents=True)
            fault = fault_root / "fault.mat"
            fault.write_bytes(f"fault-{case_id}".encode("ascii"))
            links.append(
                {
                    "GeologyId": geology_id,
                    "GeologyHash": geology_hash,
                    "Level3CaseId": case_id,
                    "PcRepresentation": "full_slice",
                    "GeologyIdVerified": "true",
                    "ReadableVerified": "true",
                    "CaseCompletionGateValidated": "true",
                    "FaultInputFolder": str(fault_root),
                    "FaultInputFile": fault.name,
                    "FileSizeBytes": fault.stat().st_size,
                }
            )
        write_csv(local_links, links)
        global_links = self.root / "geology_fault_case_links.csv"
        write_csv(global_links, links)

        manifest = self.root / "geology_stratigraphy_manifest.csv"
        write_csv(
            manifest,
            [
                {
                    "GeologyId": geology_id,
                    "GeologyHash": geology_hash,
                    "LinkedFaultCaseCount": len(CASE_IDS),
                    "StratigraphyMat": stratigraphy.relative_to(self.root),
                    "StratigraphyMatSha256": digest(stratigraphy),
                    "StratigraphyMatBytes": stratigraphy.stat().st_size,
                    "LayerSummaryCsv": summary.relative_to(self.root),
                    "FaultCaseLinkCsv": local_links.relative_to(self.root),
                }
            ],
        )

        inventory_files = [stratigraphy, summary, local_links, manifest, global_links]
        checksums = self.root / "SHA256SUMS"
        checksums.write_text(
            "".join(
                f"{digest(path)}  {path.relative_to(self.root).as_posix()}\n"
                for path in inventory_files
            ),
            encoding="utf-8",
        )
        completion = {
            "schema_version": 2,
            "status": "complete",
            "content_type": "production_geology_stratigraphy_package",
            "expected_geology_count": 1,
            "generated_geology_count": 1,
            "expected_fault_case_count": len(CASE_IDS),
            "linked_fault_case_count": len(CASE_IDS),
            "cases_per_geology": len(CASE_IDS),
            "expected_case_ids": list(CASE_IDS),
            "pc_representation": "full_slice",
            "geology_manifest_sha256": digest(manifest),
            "fault_case_links_sha256": digest(global_links),
            "checksums_sha256": digest(checksums),
        }
        (self.root / "geology_stratigraphy.done.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_verifier(self, case_ids: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--package-root",
                str(self.root),
                "--expected-geologies",
                "1",
                "--expected-cases",
                str(len(CASE_IDS)),
                "--expected-cases-per-geology",
                str(len(CASE_IDS)),
                "--expected-case-ids",
                case_ids,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_noncontiguous_case_ids(self) -> None:
        result = self.run_verifier("1,101,103")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"case_ids": [', result.stdout)

    def test_rejects_wrong_case_identity(self) -> None:
        result = self.run_verifier("1,2,103")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("case IDs", result.stderr)


if __name__ == "__main__":
    unittest.main()
