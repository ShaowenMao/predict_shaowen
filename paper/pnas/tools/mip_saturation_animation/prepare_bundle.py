"""Freeze the eight saved inputs and native curves into a portable data bundle."""
import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np

from data import geometry, read_core, sha


def prepare(a):
    a.output.mkdir(parents=True, exist_ok=False)
    raw = json.loads((a.audit / "raw.json").read_text())
    saved = json.loads((a.audit / "analysis.json").read_text())
    assert sha(a.audit / "raw.json") == saved["raw_sha256"]
    selection = json.loads(Path(__file__).with_name("selection.json").read_text())
    manifest = dict(schema=1, audit_sha256=saved["raw_sha256"], cores=[])
    for risk, selected in selection.items():
        native_path = a.native / f"{risk}_mip_wrr60_fault_nohys.mat"
        assert sha(native_path) == saved["cases"][risk]["input_sha256"]
        with h5py.File(native_path) as native:
            sgof = native["fault/fluid_tables/SGOF"]
            tables = {int(reg): native[ref][()].T for reg, ref in
                zip(sgof["regions"][()].ravel(), sgof["tables"][()].ravel())}
            regions = native["window_slice/saturation_region"][()].T
            for slot in selected["cores"]:
                key = f"{risk}_{slot:03d}"
                dest = a.output / key
                dest.mkdir()
                record = next(r for r in raw["cases"][risk] if r["slot"] == slot)
                for suffix, name, hash_key in (("", "raw.bin", "raw_input_sha256"),
                        ("_geometry", "geometry.bin", "geometry_sha256")):
                    source = a.audit / "fine_inputs" / f"{key}{suffix}.bin"
                    assert sha(source) == record["summary"][hash_key]
                    shutil.copyfile(source, dest / name)
                core = read_core(dest / "raw.bin")
                dims, spacing, extents = geometry(core, dest / "geometry.bin", sha(dest / "raw.bin"))
                w, strike = divmod(slot - 1, 87)
                table = tables[int(regions[w, strike])]
                curve = np.asarray(record["curve"])
                assert curve.shape == (100, len(record["curve_columns"]))
                assert set(np.unique(core["material"])) == {1, 2}
                assert np.isfinite(curve).all() and np.isfinite(table[:, :3]).all()
                assert np.all(np.diff(table[:, 0]) >= 0)
                for nc, rc in ((1, 3), (2, 2)):
                    np.testing.assert_allclose(np.interp(curve[:, 1], table[:, 0], table[:, nc]),
                        curve[:, rc], rtol=0, atol=1.01e-9)
                np.savetxt(dest / "native_kr.csv", table[:, :3], delimiter=",",
                    header="bulk_Sg,gas_Kr,brine_Kr", comments="")
                (dest / "record.json").write_text(json.dumps(record, indent=2) + "\n")
                files = {p.name: sha(p) for p in sorted(dest.iterdir())}
                entry = dict(key=key, risk=risk, slot=slot, case=selected["case"], window=w + 1,
                    strike_index=strike + 1, cells=core["n"], dims=list(dims), spacing=spacing,
                    physical_extent_m=extents.tolist(), files=files,
                    native_mat_sha256=sha(native_path))
                manifest["cores"].append(entry)
                print(f"Prepared {key}: {core['n']} cells, {len(curve)} saved states", flush=True)
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (a.output / "render_keys.txt").write_text("\n".join(e["key"] for e in manifest["cores"]
        if e["key"] != "high_172") + "\n", newline="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    prepare(p.parse_args())
