"""Portable, read-only geometry and saturation validation for saved MIP cores."""
import hashlib
import itertools
import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pyvista as pv

COLORS = ["#edf3f6", "#f9df9d", "#ed9a3b", "#d35121", "#9e251b"]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_core(path):
    with Path(path).open("rb") as f:
        magic, core, n, nk, ne, ni, no, n1, n2, np_, nd, nc = np.fromfile(f, "<i8", 12)
        assert magic == 2026090904 and nc == 15 and n > 0
        cfg = np.fromfile(f, "<f8", nc)
        perm = np.fromfile(f, "<f8", n * nk).reshape((n, nk), order="F")
        phi, volume = [np.fromfile(f, "<f8", n) for _ in range(2)]
        edges = np.fromfile(f, "<i8", 2 * ne).reshape((ne, 2), order="F") - 1
        inlet, outlet = [np.fromfile(f, "<i8", count) - 1 for count in (ni, no)]
        material = np.fromfile(f, "u1", n)
        tables = [np.fromfile(f, "<f8", 4 * count).reshape((count, 4), order="F") for count in (n1, n2)]
        pressures = np.fromfile(f, "<f8", np_)
        dictionary = np.fromfile(f, "<f8", nd)
        assert not f.read(1)
    return dict(core=int(core), n=int(n), cfg=cfg, perm=perm, phi=phi, volume=volume,
        edges=edges, inlet=inlet, outlet=outlet, material=material, tables=tables,
        pressures=pressures, dictionary=dictionary)


def geometry(core, path, expected_hash):
    strides = np.unique(np.abs(np.diff(core["edges"], axis=1)))
    assert len(strides) == 3 and strides[0] == 1
    nx, ny = int(strides[1]), int(strides[2] // strides[1])
    dims = (nx, ny, core["n"] // (nx * ny))
    assert np.prod(dims) == core["n"]
    ids = np.arange(core["n"]).reshape(dims, order="F")
    expected = np.concatenate([np.column_stack((
        np.take(ids, np.arange(dims[a] - 1), axis=a).ravel(),
        np.take(ids, np.arange(1, dims[a]), axis=a).ravel())) for a in range(3)])
    actual, expected = np.sort(core["edges"], axis=1), np.sort(expected, axis=1)
    np.testing.assert_array_equal(np.sort(actual[:, 0] * core["n"] + actual[:, 1]),
        np.sort(expected[:, 0] * core["n"] + expected[:, 1]))
    np.testing.assert_array_equal(np.sort(core["inlet"]), np.sort(ids[:, :, 0].ravel()))
    np.testing.assert_array_equal(np.sort(core["outlet"]), np.sort(ids[:, :, -1].ravel()))
    with Path(path).open("rb") as f:
        magic, index, axis, n, ne, ni, no = np.fromfile(f, "<i8", 7)
        length, area = np.fromfile(f, "<f8", 2)
        parent = f.read(32).hex()
        edges = np.fromfile(f, "<i8", ne * 2).reshape((ne, 2), order="F") - 1
        half = [np.fromfile(f, "<f8", ne) for _ in range(2)]
        boundaries = [np.fromfile(f, "<i8", count) - 1 for count in (ni, no)]
        boundary_half = [np.fromfile(f, "<f8", count) for count in (ni, no)]
        assert not f.read(1)
    assert (magic, index, axis, n) == (2026090906, core["core"], 3, core["n"])
    assert parent == expected_hash
    np.testing.assert_array_equal(edges, core["edges"])
    for cells, key in zip(boundaries, ("inlet", "outlet")):
        np.testing.assert_array_equal(cells, core[key])
    perm = core["perm"]
    diagonal = {1: [0, 0, 0], 3: [0, 1, 2], 6: [0, 3, 5]}[perm.shape[1]]
    spacing, residuals = [], []
    for a, stride in enumerate((1, dims[0], dims[0] * dims[1])):
        mask = edges[:, 1] - edges[:, 0] == stride
        distances = np.concatenate([np.sqrt(2 * perm[edges[mask, side], diagonal[a]]
            * core["volume"][edges[mask, side]] / half[side][mask]) for side in (0, 1)])
        d = float(np.median(distances))
        residuals.append(float(np.max(np.abs(distances / d - 1))))
        spacing.append(d)
    assert max(residuals) < 1e-8
    extents = np.asarray(dims) * spacing
    np.testing.assert_allclose(np.prod(spacing), core["volume"], rtol=1e-8, atol=0)
    np.testing.assert_allclose(extents[2], length, rtol=1e-8, atol=0)
    np.testing.assert_allclose(np.prod(extents[:2]), area, rtol=1e-8, atol=0)
    for cells, half_values in zip(boundaries, boundary_half):
        np.testing.assert_allclose(2 * perm[cells, diagonal[2]] * core["volume"][cells]
            / spacing[2] ** 2, half_values, rtol=1e-8, atol=0)
    return dims, spacing, extents


def build_grid(core, dims, spacing):
    grid = pv.ImageData(dimensions=np.asarray(dims) + 1, spacing=spacing).cast_to_structured_grid()
    ijk = np.floor(grid.cell_centers().points / np.asarray(spacing) + 1e-8).astype(int)
    np.testing.assert_array_equal(ijk, np.column_stack(np.unravel_index(np.arange(core["n"]), dims, order="F")))
    points = grid.points.copy()
    points[:, 0] *= 22
    grid.points = points
    grid.rotate_z(-60, point=grid.center, inplace=True)
    grid.cell_data["source_cell_id"] = np.arange(core["n"])
    selections = {name: grid.extract_cells(np.flatnonzero(core["material"] == code))
        for name, code in (("sand", 1), ("clay", 2))}
    assert sum(s.n_cells for s in selections.values()) == core["n"]
    for name, code in (("sand", 1), ("clay", 2)):
        np.testing.assert_array_equal(np.sort(selections[name].cell_data["source_cell_id"]),
            np.flatnonzero(core["material"] == code))
    return grid, selections


def camera_for_right_dipping_material(grid):
    center = np.asarray(grid.center)
    return [center + np.array([-.72, -1.15, .52]) * grid.length, center, (0., 0., 1.)]


def camera_scale(items):
    sizes = []
    for item in items:
        position, focus, view_up = map(np.asarray, camera_for_right_dipping_material(item["grid"]))
        view = focus - position
        view /= np.linalg.norm(view)
        right = np.cross(view, view_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, view)
        points = item["grid"].points - focus
        sizes += [float(np.ptp(points @ up)), float(np.ptp(points @ right) * 1900 / 1800)]
    return max(sizes) * .57


def local_axes_from_transformed_grid(grid):
    nx, ny, nz = grid.dimensions
    assert min(nx, ny, nz) >= 2
    result = {}
    for name, index in (("x", 1), ("y", nx), ("z", nx * ny)):
        direction = grid.points[index] - grid.points[0]
        result[name] = direction / np.linalg.norm(direction)
    return result


def outline(grid):
    dims = np.asarray(grid.dimensions)
    corners = list(itertools.product((0, 1), repeat=3))
    return np.asarray([[grid.points[np.ravel_multi_index(np.asarray(c) * (dims - 1), dims, order="F")]
        for c in (a, b)] for a, b in itertools.combinations(corners, 2)
        if sum(x != y for x, y in zip(a, b)) == 1])


def verify_arrays(core, record, hist, sg):
    curve = np.asarray(record["curve"], dtype=float)
    assert sg.shape == (len(curve), core["n"])
    assert np.isfinite(sg).all() and sg.min() >= 0 and sg.max() <= 1
    np.testing.assert_allclose(core["pressures"], curve[:, 0], rtol=1e-12, atol=1e-12)
    errors = {}
    for name, col in (("pressure_pa", 0), ("sg_physical", 1), ("trapped_pv_fraction", 4),
            ("immobile_pv_fraction", 5), ("gas_breakthrough", 6), ("invaded_pv_fraction", 7)):
        errors[name] = float(np.max(np.abs(hist[name].to_numpy() - curve[:, col])))
        np.testing.assert_allclose(hist[name], curve[:, col], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(hist["sample"], np.arange(1, len(curve) + 1))
    np.testing.assert_array_equal(hist["events"], record["summary"]["event_counts"])
    weights = core["phi"] * core["volume"]
    bulk = sg @ weights / weights.sum()
    np.testing.assert_allclose(bulk, curve[:, 1], rtol=1e-12, atol=1e-12)
    assert np.min(np.diff(sg, axis=0)) >= -1e-12
    means = np.column_stack([sg[:, core["material"] == m] @ weights[core["material"] == m]
        / weights[core["material"] == m].sum() for m in (1, 2)])
    fractions = np.array([weights[core["material"] == m].sum() / weights.sum() for m in (1, 2)])
    np.testing.assert_allclose(means @ fractions, bulk, rtol=1e-12, atol=1e-12)
    return curve, means, errors


def load_checked(bundle, key, output):
    bundle, output = Path(bundle), Path(output)
    manifest = json.loads((bundle / "manifest.json").read_text())
    entry = next(e for e in manifest["cores"] if e["key"] == key)
    source = bundle / key
    for name, expected in entry["files"].items():
        assert sha(source / name) == expected, name
    record = json.loads((source / "record.json").read_text())
    core = read_core(source / "raw.bin")
    assert core["core"] == entry["slot"]
    dims, spacing, extents = geometry(core, source / "geometry.bin", entry["files"]["raw.bin"])
    with (output / "saturation_states.bin").open("rb") as f:
        n, nt = np.fromfile(f, "<i8", 2)
        sg = np.fromfile(f, "<f8").reshape((nt, n))
    meta = tomllib.loads((output / "reconstruction.toml").read_text())
    assert meta["input_sha256"] == entry["files"]["raw.bin"] and meta["new_pressure_flow_solves"] == 0
    ref = Path(__file__).resolve().parent / "reference"
    for name, digest in meta["sources"].items():
        assert sha(ref / name) == digest, name
    curve, means, errors = verify_arrays(core, record, pd.read_csv(output / "reconstructed_history.csv"), sg)
    native = np.loadtxt(source / "native_kr.csv", delimiter=",", skiprows=1, ndmin=2)
    for native_col, raw_col in ((1, 3), (2, 2)):
        np.testing.assert_allclose(np.interp(curve[:, 1], native[:, 0], native[:, native_col]),
            curve[:, raw_col], rtol=0, atol=1.01e-9)
    grid, selections = build_grid(core, dims, spacing)
    info = dict(entry, dims=list(dims), physical_extent_m=extents.tolist())
    report = dict(status="PASS", core=entry["slot"], case=entry["case"], risk=entry["risk"],
        cells=int(n), states=int(nt), saved_history_max_abs_errors=errors,
        raw_input_sha256=entry["files"]["raw.bin"], geometry_sha256=entry["files"]["geometry.bin"],
        saved_audit_sha256=manifest["audit_sha256"], native_kr_sha256=entry["files"]["native_kr.csv"],
        state_sha256=sha(output / "saturation_states.bin"), dimensions=list(dims),
        physical_extent_m=extents.tolist(), thickness_exaggeration=22,
        reconstructed_not_archived_cell_states=True, new_flow_solves=0,
        production_files_changed=False, event_counts_match=True,
        kr_source="Original accepted 100-state curve; no new pressure-flow solve",
        animation_axis="Increasing applied gas-brine pressure difference, not elapsed physical time",
        native_table_max_abs_error=float(max(np.max(abs(np.interp(curve[:, 1], native[:, 0], native[:, col])
            - curve[:, raw_col])) for col, raw_col in ((1, 3), (2, 2)))))
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    frame_table = pd.DataFrame(curve, columns=record["curve_columns"])
    frame_table.insert(0, "state", np.arange(1, nt + 1))
    frame_table["sand_mean_sg"], frame_table["clay_mean_sg"] = means.T
    frame_table.to_csv(output / "frame_values.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    return dict(info=info, grid=grid, selections=selections), curve, sg, means
