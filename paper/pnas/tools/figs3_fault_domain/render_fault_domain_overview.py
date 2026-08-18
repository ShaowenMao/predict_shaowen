"""Render a Step62 fault-domain schematic following the SI Fig. 1(c) workflow.

The complete listric fault domain is colored by its host interval: storage
reservoir, top seal, and overburden.  The six top-seal throw windows and 87
along-strike segments are delineated on the visible fault face.  Text
annotations are deliberately deferred to a later composition pass.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv


HERE = Path(__file__).resolve().parent
V1_SCRIPT = HERE / "render_fault_domain_base.py"
DEFAULT_OUTPUT = (
    HERE.parents[1] / "figures" / "source" / "figs3_fault_overview_vector_base.png"
)


def load_v1_module():
    spec = importlib.util.spec_from_file_location("fault_domain_fig1c_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(V1_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_v1_module()

# Use the established SI Fig. 1(c) geological palette.
OVERBURDEN_COLOR = "#C6B06E"
TOP_SEAL_COLOR = "#5F382D"
STORAGE_RESERVOIR_COLOR = "#D59B47"
STRUCTURE_COLOR = "#173B5E"
SEGMENT_COLOR = "#D8B58C"
WINDOW_BOUNDARY_COLOR = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, default=BASE.DEFAULT_GRID_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=10_000)
    parser.add_argument("--height", type=int, default=6_250)
    parser.add_argument("--vertical-exaggeration", type=float, default=3.0)
    parser.add_argument(
        "--camera",
        type=float,
        nargs=3,
        metavar=("ALONG_STRIKE", "FAULT_NORMAL", "ELEVATION"),
        # Exact camera direction used by the established multipart-workflow
        # fault panel (render_3d_fault_geometry.py).
        default=(1.50, -1.00, 0.00),
    )
    parser.add_argument("--camera-zoom", type=float, default=1.12)
    return parser.parse_args()


def boundary_edges(
    triangles: np.ndarray, selected_cells: np.ndarray
) -> list[tuple[int, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for triangle in triangles[np.asarray(selected_cells, dtype=np.int64)]:
        for first, second in zip(triangle, np.roll(triangle, -1)):
            counts[tuple(sorted((int(first), int(second))))] += 1
    return [edge for edge, count in counts.items() if count == 1]


def near_side_profile(
    yz: np.ndarray,
    exterior_edges: list[tuple[int, int]],
    sample_count: int = 180,
) -> np.ndarray:
    """Sample the smaller-y exterior boundary of a selected fault region."""
    edge_array = np.asarray(exterior_edges, dtype=np.int64)
    depth_min = float(yz[edge_array.ravel(), 1].min())
    depth_max = float(yz[edge_array.ravel(), 1].max())
    depth_samples = np.linspace(depth_min + 1.0e-6, depth_max - 1.0e-6, sample_count)
    fault_normal_samples: list[float] = []
    for depth in depth_samples:
        intersections: list[float] = []
        for first, second in exterior_edges:
            point_a = yz[first]
            point_b = yz[second]
            depth_a, depth_b = point_a[1], point_b[1]
            if np.isclose(depth_a, depth_b):
                continue
            if min(depth_a, depth_b) <= depth <= max(depth_a, depth_b):
                fraction = (depth - depth_a) / (depth_b - depth_a)
                intersections.append(
                    float(point_a[0] + fraction * (point_b[0] - point_a[0]))
                )
        if len(intersections) < 2:
            raise RuntimeError(f"Selected fault boundary is open at depth={depth:g} m")
        fault_normal_samples.append(min(intersections))
    return np.column_stack((fault_normal_samples, depth_samples))


def strike_segment_lines(
    profile: np.ndarray,
    x_planes: np.ndarray,
    vertical_exaggeration: float,
) -> pv.PolyData:
    """Build one down-dip line at each exact along-strike segment boundary."""
    point_count = profile.shape[0]
    points = np.empty((x_planes.size * point_count, 3), dtype=float)
    lines: list[int] = []
    for index, x_coordinate in enumerate(x_planes):
        start = index * point_count
        points[start : start + point_count, 0] = x_coordinate
        points[start : start + point_count, 1] = profile[:, 0]
        points[start : start + point_count, 2] = (
            -vertical_exaggeration * profile[:, 1]
        )
        lines.extend([point_count, *range(start, start + point_count)])
    return pv.PolyData(points, lines=np.asarray(lines, dtype=np.int64))


def throw_window_interface_points(
    yz: np.ndarray,
    triangles: np.ndarray,
    groups: dict[int, np.ndarray],
    full_fault_exterior_edges: list[tuple[int, int]],
) -> np.ndarray:
    """Find the near-side endpoints of boundaries enclosing W1--W6."""
    cell_group: dict[int, int] = {}
    for group, cell_ids in groups.items():
        for cell_id in cell_ids:
            cell_group[int(cell_id)] = group

    edge_groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for cell_id, group in cell_group.items():
        triangle = triangles[cell_id]
        for first, second in zip(triangle, np.roll(triangle, -1)):
            edge_groups[tuple(sorted((int(first), int(second))))].add(group)

    exterior_nodes = set(np.asarray(full_fault_exterior_edges).ravel().tolist())
    points: list[np.ndarray] = []
    # Interfaces 5/6 and 11/12 enclose the PREDICT interval; 6/7 through
    # 10/11 are the five internal boundaries between its six throw windows.
    for lower_group in range(5, 12):
        upper_group = lower_group + 1
        interface_nodes: set[int] = set()
        for edge, neighboring_groups in edge_groups.items():
            if neighboring_groups == {lower_group, upper_group}:
                interface_nodes.update(edge)
        candidates = sorted(interface_nodes.intersection(exterior_nodes))
        if not candidates:
            raise RuntimeError(
                f"No exterior endpoint for UCID interface {lower_group}/{upper_group}"
            )
        points.append(yz[min(candidates, key=lambda node: yz[node, 0])])
    return np.asarray(points)


def throw_window_lines(
    interface_yz: np.ndarray,
    x_min: float,
    x_max: float,
    vertical_exaggeration: float,
) -> pv.PolyData:
    points: list[list[float]] = []
    lines: list[int] = []
    for fault_normal, depth in interface_yz:
        start = len(points)
        points.extend(
            [
                [x_min, fault_normal, -vertical_exaggeration * depth],
                [x_max, fault_normal, -vertical_exaggeration * depth],
            ]
        )
        lines.extend([2, start, start + 1])
    return pv.PolyData(np.asarray(points), lines=np.asarray(lines, dtype=np.int64))


def full_fault_outline(fault: pv.UnstructuredGrid) -> pv.PolyData:
    return fault.extract_surface(algorithm="dataset_surface").extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        feature_angle=18.0,
        manifold_edges=False,
        non_manifold_edges=False,
    )


def render(args: argparse.Namespace) -> None:
    if args.vertical_exaggeration <= 0.0:
        raise ValueError("vertical exaggeration must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    yz, triangles, groups = BASE.load_step62_fault(args.grid_dir)
    x_planes = np.r_[0.0, np.cumsum(BASE.along_strike_widths())]
    fault = BASE.build_extruded_fault(
        yz,
        triangles,
        groups,
        x_planes,
        args.vertical_exaggeration,
    )
    storage_reservoir = fault.threshold(
        value=(4.5, 5.5), scalars="fault_group", preference="cell"
    ).extract_surface(algorithm="dataset_surface").clean()
    top_seal = fault.threshold(
        value=(5.5, 11.5), scalars="fault_group", preference="cell"
    ).extract_surface(algorithm="dataset_surface").clean()
    overburden = fault.threshold(
        value=(11.5, 13.5), scalars="fault_group", preference="cell"
    ).extract_surface(algorithm="dataset_surface").clean()

    full_fault_cells = np.unique(np.concatenate(tuple(groups.values())))
    predict_cells = np.unique(
        np.concatenate(tuple(groups[group] for group in BASE.PREDICT_GROUPS))
    )
    full_exterior = boundary_edges(triangles, full_fault_cells)
    predict_exterior = boundary_edges(triangles, predict_cells)
    profile = near_side_profile(yz, predict_exterior)
    segment_lines = strike_segment_lines(
        profile, x_planes, args.vertical_exaggeration
    )
    interfaces = throw_window_interface_points(
        yz, triangles, groups, full_exterior
    )
    window_lines = throw_window_lines(
        interfaces,
        float(x_planes[0]),
        float(x_planes[-1]),
        args.vertical_exaggeration,
    )
    camera_vector = np.asarray(args.camera, dtype=float)
    if np.linalg.norm(camera_vector) == 0.0:
        raise ValueError("camera vector cannot be zero")
    # Move line overlays 120 m toward the camera so they remain legible and
    # free of z-fighting without visibly changing their location at 45-km scale.
    line_offset = 120.0 * camera_vector / np.linalg.norm(camera_vector)
    segment_lines.translate(line_offset, inplace=True)
    window_lines.translate(line_offset, inplace=True)

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.width, args.height),
        lighting="three lights",
    )
    plotter.set_background("white", top="white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

    for surface, color in (
        (overburden, OVERBURDEN_COLOR),
        (top_seal, TOP_SEAL_COLOR),
        (storage_reservoir, STORAGE_RESERVOIR_COLOR),
    ):
        plotter.add_mesh(
            surface,
            color=color,
            opacity=1.0,
            smooth_shading=False,
            lighting=False,
        )
    plotter.add_mesh(
        full_fault_outline(fault),
        color=STRUCTURE_COLOR,
        opacity=1.0,
        line_width=2.8,
        render_lines_as_tubes=True,
        lighting=False,
    )
    plotter.add_mesh(
        segment_lines,
        color=SEGMENT_COLOR,
        opacity=1.0,
        line_width=3.2,
        render_lines_as_tubes=True,
        lighting=False,
    )
    plotter.add_mesh(
        window_lines,
        color=WINDOW_BOUNDARY_COLOR,
        opacity=1.0,
        line_width=4.0,
        render_lines_as_tubes=True,
        lighting=False,
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
        center + distance * camera_vector,
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(args.camera_zoom)

    plotter.show(auto_close=False)
    try:
        plotter.screenshot(
            str(args.output),
            transparent_background=True,
            return_img=False,
        )
    finally:
        plotter.close()
    BASE.transparent_crop(args.output)

    print(f"Fault cells: {fault.n_cells:,}")
    print("Throw windows: 6")
    print(f"Along-strike segments per window: {x_planes.size - 1}")
    print(f"Vertical exaggeration: {args.vertical_exaggeration:g}x")
    print(args.output)


if __name__ == "__main__":
    render(parse_args())
