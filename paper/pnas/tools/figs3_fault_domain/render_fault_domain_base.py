"""Render a clean Step62 fault-domain base using the SI Fig. 1(c) style.

This first iterative candidate intentionally contains no labels, legend, grid
lines, or annotations.  It establishes the camera, vertical exaggeration,
lighting, material colors, and transparent-background workflow before later
figure elements are added.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
DEFAULT_GRID_DIR = (
    HERE.parents[4]
    / "mrst_predict_sim_grid_integration"
    / "setup_shaowen_resolution"
    / "grid_candidates"
    / "step_62_matched_upper_lower_transition"
)
DEFAULT_OUTPUT = HERE / "step62_fault_domain_base.png"

# Match the established SI Fig. 1(c) renderer.
PREDICT_COLOR = "#2C8C99"
NONPREDICT_COLOR = "#173B5E"
VERTICAL_EXAGGERATION = 1.5
CAMERA_VECTOR = np.asarray([1.00, -1.15, 0.56], dtype=float)

# Raw Step62 UCID groups: storage-reservoir fault, W1--W6, MM--UM, Younger.
FAULT_GROUPS = tuple(range(5, 14))
PREDICT_GROUPS = tuple(range(6, 12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2600)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument(
        "--vertical-exaggeration", type=float, default=VERTICAL_EXAGGERATION
    )
    return parser.parse_args()


def along_strike_widths() -> np.ndarray:
    """Return the exact 87 Step62 along-strike segment widths in metres."""
    half = (
        [100.0]
        + [150.0] * 28
        + [175.0, 200.0, 250.0, 300.0, 400.0, 650.0, 1000.0]
        + list(np.arange(1600.0, 2600.1, 200.0))
        + [2600.0]
    )
    widths = np.asarray(list(reversed(half)) + [50.0] + half)
    if widths.size != 87 or not np.isclose(widths.sum(), 45_000.0):
        raise ValueError("Unexpected Step62 along-strike discretization")
    return widths


def load_step62_fault(
    grid_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Load the protected Step62 cross-section and fault UCID groups."""
    nodes_path = grid_dir / "nodes_coordinates.dat"
    triangles_path = grid_dir / "t.mat"
    ucids_path = grid_dir / "ucids_sc2_2D.mat"
    for path in (nodes_path, triangles_path, ucids_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    yz = np.loadtxt(nodes_path)
    yz[:, 1] *= -1.0  # source elevation -> positive-downward depth
    triangles = loadmat(triangles_path, squeeze_me=True)["t"].astype(np.int64) - 1
    unit_cell_ids = loadmat(
        ucids_path,
        squeeze_me=True,
        struct_as_record=False,
    )["unit_cell_ids"]
    groups = {
        group: np.atleast_1d(unit_cell_ids[group - 1]).astype(np.int64) - 1
        for group in FAULT_GROUPS
    }

    counts = [groups[group].size for group in PREDICT_GROUPS]
    if counts != [62, 63, 62, 64, 62, 57]:
        raise ValueError(f"Unexpected W1--W6 cell counts: {counts}")
    if sum(ids.size for ids in groups.values()) != 1_731:
        raise ValueError("Expected 1,731 Step62 fault triangles per cross-section")
    return yz, triangles, groups


def build_extruded_fault(
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
    x_planes: np.ndarray,
    vertical_exaggeration: float,
) -> pv.UnstructuredGrid:
    """Extrude the exact 2-D fault footprint into the 3-D Step62 fault domain."""
    fault_cells = np.unique(np.concatenate(tuple(groups.values())))
    fault_nodes = np.unique(triangles[fault_cells].ravel())
    old_to_local = np.full(yz.shape[0], -1, dtype=np.int64)
    old_to_local[fault_nodes] = np.arange(fault_nodes.size)
    local_yz = yz[fault_nodes]

    n_nodes = local_yz.shape[0]
    points = np.empty((n_nodes * x_planes.size, 3), dtype=float)
    for plane, x_coordinate in enumerate(x_planes):
        block = slice(plane * n_nodes, (plane + 1) * n_nodes)
        points[block, 0] = x_coordinate
        points[block, 1] = local_yz[:, 0]
        points[block, 2] = -vertical_exaggeration * local_yz[:, 1]

    cells: list[np.ndarray] = []
    class_values: list[np.ndarray] = []
    window_values: list[np.ndarray] = []
    group_values: list[np.ndarray] = []
    for group in FAULT_GROUPS:
        local_triangles = old_to_local[triangles[groups[group]]]
        for segment in range(x_planes.size - 1):
            lower = local_triangles + segment * n_nodes
            upper = local_triangles + (segment + 1) * n_nodes
            wedges = np.column_stack((lower, upper))
            cells.append(np.column_stack((np.full(wedges.shape[0], 6), wedges)))
            is_predict = int(group in PREDICT_GROUPS)
            class_values.append(np.full(wedges.shape[0], is_predict, dtype=np.int8))
            window_values.append(
                np.full(
                    wedges.shape[0],
                    group - 5 if is_predict else 0,
                    dtype=np.int8,
                )
            )
            group_values.append(
                np.full(wedges.shape[0], group, dtype=np.int8)
            )

    packed_cells = np.vstack(cells)
    cell_types = np.full(packed_cells.shape[0], pv.CellType.WEDGE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(packed_cells.ravel(), cell_types, points)
    grid.cell_data["predict_region"] = np.concatenate(class_values)
    grid.cell_data["throw_window"] = np.concatenate(window_values)
    grid.cell_data["fault_group"] = np.concatenate(group_values)
    if grid.n_cells != 150_597:
        raise ValueError(f"Expected 150,597 fault cells, found {grid.n_cells:,}")
    return grid


def transparent_crop(path: Path, padding: int = 26) -> None:
    """Crop transparent margins while preserving antialiased object edges."""
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA")).copy()
        rgb = rgba[:, :, :3].astype(float)
        # Handle Windows VTK backends that return opaque white screenshots.
        distance_from_white = 255.0 - np.min(rgb, axis=2)
        rgba[:, :, 3] = np.clip(
            5.0 * distance_from_white, 0.0, 255.0
        ).astype(np.uint8)
        image = Image.fromarray(rgba)
        bounds = image.getchannel("A").getbbox()
        if bounds is None:
            raise RuntimeError("VTK produced an empty image")
        left, top, right, bottom = bounds
        image.crop(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width, right + padding),
                min(image.height, bottom + padding),
            )
        ).save(path, dpi=(600, 600), optimize=True)


def render(args: argparse.Namespace) -> None:
    if args.vertical_exaggeration <= 0.0:
        raise ValueError("vertical exaggeration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    yz, triangles, groups = load_step62_fault(args.grid_dir)
    x_planes = np.r_[0.0, np.cumsum(along_strike_widths())]
    fault = build_extruded_fault(
        yz,
        triangles,
        groups,
        x_planes,
        args.vertical_exaggeration,
    )
    nonpredict = fault.threshold(
        value=(-0.25, 0.25), scalars="predict_region", preference="cell"
    ).extract_surface(algorithm="dataset_surface").clean()
    predict = fault.threshold(
        value=(0.75, 1.25), scalars="predict_region", preference="cell"
    ).extract_surface(algorithm="dataset_surface").clean()

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.width, args.height),
        lighting="three lights",
    )
    plotter.set_background("white", top="white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.add_mesh(
        nonpredict,
        color=NONPREDICT_COLOR,
        opacity=1.0,
        smooth_shading=True,
        ambient=0.42,
        diffuse=0.65,
        specular=0.0,
    )
    plotter.add_mesh(
        predict,
        color=PREDICT_COLOR,
        opacity=1.0,
        smooth_shading=True,
        ambient=0.38,
        diffuse=0.72,
        specular=0.0,
    )

    bounds = np.asarray(fault.bounds, dtype=float)
    center = np.asarray(
        [
            np.mean(bounds[0:2]),
            np.mean(bounds[2:4]),
            np.mean(bounds[4:6]),
        ]
    )
    extent = np.asarray(
        [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    )
    distance = float(np.linalg.norm(extent))
    plotter.camera_position = [
        center + distance * CAMERA_VECTOR,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.10)

    plotter.show(auto_close=False)
    try:
        plotter.screenshot(
            str(args.output),
            transparent_background=True,
            return_img=False,
        )
    finally:
        plotter.close()
    transparent_crop(args.output)

    print(f"Fault cells: {fault.n_cells:,}")
    print(f"Along-strike segments: {x_planes.size - 1}")
    print(f"Vertical exaggeration: {args.vertical_exaggeration:g}x")
    print(args.output)


if __name__ == "__main__":
    render(parse_args())
