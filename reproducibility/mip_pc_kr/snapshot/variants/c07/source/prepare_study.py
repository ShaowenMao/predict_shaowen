"""Stage one isolated comparison; never submit or modify production files."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tomllib

CASE = "s03_c022_case12"
TASK = 1137
EXPECTED_SPECIFIC = "82fac5cd1e40a65ee26167dc6e54224b469ee84adce5c03148082c0dd7c14c71"
EXPECTED_TABLES = "b88dd2d8a7b9d4f612611f88e35c3ce6c93a94313a4877e513c169ad18a9c481"
EXPECTED_BASELINE = "adf73c15b1c4edc4b4721c7666a2c2b56aaf800088fff3599a5541dce0ed3dde"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def toml_section(values):
    def value(v):
        return str(v).lower() if isinstance(v, bool) else json.dumps(v)
    return "\n".join(f"{k} = {value(v)}" for k, v in values.items()) + "\n"


def main():
    p = argparse.ArgumentParser()
    for key in ("root", "repo", "baseline", "export"):
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    root = a.root.absolute()
    assert str(root).startswith("/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/")
    assert not root.exists(), f"Refusing to overwrite {root}"
    assert sha(a.baseline) == EXPECTED_BASELINE
    spec = importlib.util.spec_from_file_location(
        "baseline_resolver", a.repo / "scripts/engaging/gom_step62_production_manifest.py")
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)
    # Validate the original schema without rehashing 2,430 unrelated input files.
    baseline = resolver.load_and_validate(str(a.baseline), verify_files=False)
    selected = copy.deepcopy(baseline["cases"][TASK - 1])
    assert selected["task"] == TASK and selected["case_key"] == CASE
    original_specific = Path(selected["specific_path"])
    candidate = a.export / f"candidate_specific_{CASE}.mat"
    tables = a.export / f"candidate_tables_{CASE}.mat"
    assert sha(candidate) == EXPECTED_SPECIFIC and sha(tables) == EXPECTED_TABLES
    assert sha(original_specific) == selected["specific_sha256"]
    assert subprocess.check_output(["git", "-C", str(a.repo), "rev-parse", "HEAD"], text=True).strip() == baseline["jutuldarcy_commit"]
    assert sha(a.repo / "Manifest.toml") == baseline["jutul_manifest_sha256"]
    receipt_paths = [a.export / "runtime_validation/constitutive_pass.toml",
                     a.export / "runtime_validation/full_fault_contract_pass.toml",
                     a.export / "smoke_validation_v3/PASS.toml"]
    receipts = [tomllib.loads(x.read_text()) for x in receipt_paths]
    assert all(x["passed"] for x in receipts)
    assert receipts[0]["table_sha256"] == receipts[2]["table_sha256"] == EXPECTED_TABLES
    assert receipts[1]["candidate_specific_sha256"] == EXPECTED_SPECIFIC
    assert receipts[1]["common_sha256"] == baseline["common_sha256"]
    assert receipts[2]["smoke_simulations"] == 14
    assert receipts[2]["constitutive_receipt_sha256"] == sha(receipt_paths[0])
    assert receipts[2]["full_fault_receipt_sha256"] == sha(receipt_paths[1])
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    (root / "tmp").mkdir()
    source = root / "source"
    shutil.copytree(Path(__file__).parent, source)
    for f in source.rglob("*"):
        if f.is_file():
            f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    provenance = root / "provenance"
    provenance.mkdir()
    shutil.copy2(a.baseline, provenance / "baseline_campaign.toml")
    for i, f in enumerate(receipt_paths):
        shutil.copy2(f, provenance / f"receipt_{i + 1}.toml")
    m = copy.deepcopy(baseline)
    m["campaign_id"] = "candidate_reservoir_pair_s03_c022_case12_20260910_v1"
    m["ensemble_kind"] = "isolated_candidate_pair"
    m["archive_root"] = str(root)
    m["workflow"]["archive_shard_size"] = 1
    m["workflow"]["archive_compact_atomic_rows"] = False
    m["workflow"]["archive_remove_verified_scratch"] = False
    selected.update(specific_path=str(candidate), specific_sha256=EXPECTED_SPECIFIC,
                    specific_bytes=candidate.stat().st_size)
    manifest = root / "campaign.toml"
    header = {"schema_version": 3, "candidate_manifest_schema": "isolated_case_pair_v1",
              "campaign_id": m["campaign_id"], "ensemble_kind": m["ensemble_kind"],
              "case_count": 1, "allowed_original_task": TASK, "archive_root": str(root),
              "production_promotion": False, "restart_reuse": False,
              "jutuldarcy_commit": m["jutuldarcy_commit"],
              "jutul_manifest_sha256": m["jutul_manifest_sha256"]}
    manifest.write_text(toml_section(header) + "\n[baseline]\n" + toml_section({
        "manifest": str(a.baseline), "manifest_sha256": EXPECTED_BASELINE,
        "case_order_sha256": m["case_order_sha256"],
        "specific_path": str(original_specific),
        "specific_sha256": baseline["cases"][TASK - 1]["specific_sha256"],
        "source_input_manifest_sha256": m["source_input_manifest_sha256"]}) +
        "\n[workflow]\n" + toml_section(m["workflow"]) + "\n[common]\n" +
        toml_section({"path": m["common_path"], "sha256": m["common_sha256"],
                      "bytes": m["common_bytes"]}) + "\n[[cases]]\n" + toml_section(selected))
    m.update(manifest_path=str(manifest), manifest_companion_path=str(manifest)+".sha256",
             manifest_sha256=sha(manifest))
    Path(str(manifest)+".sha256").write_text(sha(manifest)+"  campaign.toml\n")
    vals = resolver.base_shell_values(m)
    vals.update({"GOM_PRODUCTION_CASE_COUNT": 1, "GOM_PRODUCTION_ARCHIVE_SHARD_COUNT": 1,
                 "GOM_PRODUCTION_TASK": TASK, "GOM_PRODUCTION_CASE_KEY": CASE,
                 "GOM_PRODUCTION_GEOLOGY_ID": selected["geology_id"],
                 "GOM_PRODUCTION_REALIZATION_ID": selected["realization_id"],
                 "GOM_PRODUCTION_COMMON_MAT": m["common_path"],
                 "GOM_PRODUCTION_COMMON_SHA256": m["common_sha256"],
                 "GOM_PRODUCTION_COMMON_BYTES": m["common_bytes"],
                 "GOM_PRODUCTION_SPECIFIC_MAT": str(candidate),
                 "GOM_PRODUCTION_SPECIFIC_SHA256": EXPECTED_SPECIFIC,
                 "GOM_PRODUCTION_SPECIFIC_BYTES": candidate.stat().st_size,
                 "GOM_PRODUCTION_GEOLOGY_HASH": selected["geology_hash"],
                 "GOM_PRODUCTION_LEVEL3_CASE_NAME": selected["level3_case_name"],
                 "BASELINE_SPECIFIC_MAT": str(original_specific),
                 "BASELINE_MANIFEST_SHA256": EXPECTED_BASELINE,
                 "CANDIDATE_TABLES_MAT": str(tables)})
    (root / "locked.env").write_text("\n".join(
        f"export {k}={shlex.quote(str(v))}" for k, v in vals.items()) + "\n")
    inputs = {Path(m["common_path"]): m["common_sha256"], candidate: EXPECTED_SPECIFIC,
              original_specific: baseline["cases"][TASK - 1]["specific_sha256"],
              tables: EXPECTED_TABLES, a.repo / "Manifest.toml": m["jutul_manifest_sha256"],
              a.baseline: EXPECTED_BASELINE}
    inputs.update({f: sha(f) for f in receipt_paths})
    (root / "INPUTS_SHA256.txt").write_text("".join(f"{h}  {f}\n" for f,h in inputs.items()))
    protected = [manifest, Path(str(manifest)+".sha256"), root/"locked.env", root/"INPUTS_SHA256.txt"]
    protected += sorted(f for f in source.rglob("*") if f.is_file())
    (root / "STUDY_SHA256.txt").write_text("".join(
        f"{sha(f)}  {f.relative_to(root)}\n" for f in protected))
    print(json.dumps({"study":str(root), "case":CASE, "task":TASK,
                      "manifest_sha256":sha(manifest), "staged_only":True}, indent=2))


if __name__ == "__main__":
    main()
