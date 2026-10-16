"""One independent rendering task. Refuses an existing output; never submits jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data import sha


def run(a):
    workflow = Path(__file__).resolve().parent
    source = json.loads((workflow / "reference/SOURCE.json").read_text())
    for name, digest in source["files"].items():
        assert sha(workflow / "reference" / name) == digest, name
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    entry = next(e for e in manifest["cores"] if e["key"] == a.core)
    for name, digest in entry["files"].items():
        assert sha(a.bundle / a.core / name) == digest, name
    a.output.mkdir(parents=True, exist_ok=False)
    timings = {}
    def stage(name, args):
        started = time.monotonic()
        print("START " + name, flush=True)
        subprocess.run(args, check=True)
        timings[name] = time.monotonic() - started
        (a.output / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    stage("reconstruct", ["julia", "--startup-file=no", str(workflow / "reconstruct.jl"),
        str(a.bundle / a.core / "raw.bin"), entry["files"]["raw.bin"], str(a.output)])
    common = ["--bundle", str(a.bundle), "--core", a.core, "--output", str(a.output)]
    stage("validate_and_render", [sys.executable, "-B", str(workflow / "render.py"), "render", *common])
    stage("encode", [sys.executable, "-B", str(workflow / "render.py"), "encode", *common])
    stage("media_qa", [sys.executable, "-B", str(workflow / "check_media.py"), "--output", str(a.output)])
    receipt = dict(status="PASS", core=a.core, job_id=os.environ.get("SLURM_JOB_ID"), timings=timings,
        workflow_sources={str(path.relative_to(workflow)): sha(path) for path in sorted(workflow.rglob("*"))
            if path.is_file() and path.suffix in (".py", ".jl", ".sh", ".json", ".txt")
            and "__pycache__" not in path.parts},
        files={name: sha(a.output / name) for name in ("synchronized_drainage_4k.mp4", "looping_preview.gif",
            "validation.json", "media_qa.json", "frame_values.csv")})
    (a.output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--core", required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
