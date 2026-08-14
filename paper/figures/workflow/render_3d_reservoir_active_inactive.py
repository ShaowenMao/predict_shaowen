"""Render the full Step 62 reservoir with active and inactive grid volumes.

The production VTU contains only simulation-active cells. The original raw
two-dimensional mesh also retains the cells removed by ``buildGoMMesh`` as
inactive regions 56--58. This script extrudes those inactive triangles through
the same 87 along-strike intervals as the VTU and renders them as a translucent
context around the active reservoir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIVE_VTU = Path(
    r"D:\codex_gom\step62_effective_pc_global_plateau"
    r"\case5_s03_c001_case04_geology_v2"
    r"\gom_step62_effective_pc_global_plateau_"
    r"s03_c001_case04_geology_v2_0210.vtu"
)
DEFAULT_RAW_GRID = Path(
    r"D:\Github\mrst_predict_sim_grid_dev\tmp\cluster_packages"
    r"\step62_87slice_full_prepare_v1\setup_shaowen_resolution"
    r"\grid_candidates\step_62_matched_upper_lower_transition"
)
DEFAULT_FAULT_NODES = Path(
    r"D:\Github\mrst_predict_sim_grid_dev"
    r"\setup_shaowen_resolution\fnodcoord.mat"
)
DEFAULT_OUTPUT = (
    ROOT / "assets" / "derived" / "reservoir_active_inactive_3d.png"
)

ROCK_REGION = "rock_region"
FAULT_REGION = "fault_region_flag"
STRATIGRAPHIC_UNIT = "stratigraphic_unit_id"
STRATIGRAPHY_REGION = "stratigraphy_region_flag"
GEOLOGIC_REGION = "geologic_region"
INACTIVE_REGION_IDS = range(56, 59)

OVERBURDEN_COLOR = "#C6B06E"
TOP_SEAL_COLOR = "#5F382D"
STORAGE_RESERVOIR_COLOR = "#D59B47"
UNDERBURDEN_COLOR = "#A8B3BF"
PREDICT_FAULT_COLOR = "#2C8C99"
OTHER_FAULT_COLOR = "#173B5E"
SECONDARY_FAULT_COLOR = "#7B4A64"
INJECTOR_COLOR = "#D83B72"
SAND_INTERBED_COLOR = "#E4C56A"
CLAY_INTERBED_COLOR = "#8A5744"


def parse_args() -> argparse.Namespace:
    """Parse source, output, and rendering options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-vtu", type=Path, default=DEFAULT_ACTIVE_VTU)
    parser.add_argument("--raw-grid", type=Path, default=DEFAULT_RAW_GRID)
    parser.add_argument(
        "--fault-node-coordinates", type=Path, default=DEFAULT_FAULT_NODES
    )
    parser.add_argument(
        "--secondary-fault-trace",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing y_m and depth_m anchors for a second "
            "structural fault surface."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2600)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument("--vertical-exaggeration", type=float, default=1.5)
    parser.add_argument("--active-opacity", type=float, default=1.0)
    parser.add_argument("--inactive-opacity", type=float, default=0.34)
    parser.add_argument(
        "--overburden-opacity",
        type=float,
        default=None,
        help="Optional opacity override for the overburden exterior surfaces.",
    )
    parser.add_argument(
        "--top-seal-opacity",
        type=float,
        default=None,
        help="Optional opacity override for the complete top-seal package.",
    )
    parser.add_argument(
        "--storage-reservoir-opacity",
        type=float,
        default=None,
        help="Optional opacity override for the storage-reservoir exterior.",
    )
    parser.add_argument(
        "--fault-region-opacity",
        type=float,
        default=1.0,
        help="Opacity of the PREDICT and non-PREDICT fault-region surfaces.",
    )
    parser.add_argument(
        "--top-surface-opacity",
        type=float,
        default=None,
        help=(
            "Optional opacity for only the horizontal top boundary. The "
            "remaining active exterior surfaces retain --active-opacity."
        ),
    )
    parser.add_argument(
        "--opaque-front-xz",
        action="store_true",
        help=(
            "Overlay the complete near-side constant-y (x-z) exterior boundary "
            "as opaque while leaving every other model surface at its requested "
            "opacity."
        ),
    )
    parser.add_argument(
        "--opaque-active-sides",
        action="store_true",
        help=(
            "Render all four vertical exterior sides of the active domain as "
            "opaque without changing the underburden or horizontal roof."
        ),
    )
    parser.add_argument(
        "--cutaway-y",
        type=float,
        default=12250.0,
        help="Remove the near-side volume below this fault-normal coordinate.",
    )
    parser.add_argument("--camera-along-strike", type=float, default=1.00)
    parser.add_argument("--camera-fault-normal", type=float, default=-1.15)
    parser.add_argument("--camera-elevation", type=float, default=0.56)
    parser.add_argument(
        "--delineate-full-fault",
        action="store_true",
        help="Render the complete listric fault from its source-node curve.",
    )
    parser.add_argument("--full-fault-opacity", type=float, default=0.28)
    parser.add_argument("--secondary-fault-opacity", type=float, default=0.38)
    parser.add_argument(
        "--injector",
        type=float,
        nargs=3,
        metavar=("X_M", "Y_M", "DEPTH_M"),
        default=None,
        help="Optional injector centroid in global x, y, positive-down depth (m).",
    )
    parser.add_argument(
        "--injector-point-size",
        type=float,
        default=18.0,
        help="Screen-space diameter (pixels) of the schematic injector marker.",
    )
    parser.add_argument(
        "--show-injector-label",
        action="store_true",
        help="Add a text label beside the optional injector marker.",
    )
    parser.add_argument(
        "--show-legend",
        action="store_true",
        help="Include a legend for standalone diagnostic use.",
    )
    return parser.parse_args()


def inactive_triangle_ids(raw_grid: Path, triangle_count: int) -> np.ndarray:
    """Return zero-based raw triangle IDs assigned to inactive regions."""
    ucids = loadmat(
        raw_grid / "ucids_sc2_2D.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["unit_cell_ids"].ravel()
    inactive: set[int] = set()
    for region_id in INACTIVE_REGION_IDS:
        values = np.asarray(ucids[region_id - 1]).reshape(-1)
        inactive.update(
            int(value) - 1
            for value in values
            if 1 <= int(value) <= triangle_count
        )
    return np.asarray(sorted(inactive), dtype=np.int64)


def build_inactive_grid(raw_grid: Path, x_planes: np.ndarray) -> pv.UnstructuredGrid:
    """Extrude inactive raw triangles through the production layer planes."""
    vertices = np.loadtxt(raw_grid / "nodes_coordinates.dat")
    triangles = loadmat(raw_grid / "t.mat")["t"].astype(np.int64) - 1
    inactive_ids = inactive_triangle_ids(raw_grid, triangles.shape[0])
    inactive_triangles = triangles[inactive_ids]

    used_nodes = np.unique(inactive_triangles)
    raw_to_local = np.full(vertices.shape[0], -1, dtype=np.int64)
    raw_to_local[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
    local_triangles = raw_to_local[inactive_triangles]

    # buildGoMMesh maps raw 2-D coordinates to final (y, z) as (x, -y),
    # while the layer-extrusion coordinate becomes final x.
    yz = np.column_stack((vertices[used_nodes, 0], -vertices[used_nodes, 1]))
    nodes_per_plane = used_nodes.size
    points = np.empty((x_planes.size * nodes_per_plane, 3), dtype=np.float64)
    points[:, 0] = np.repeat(x_planes, nodes_per_plane)
    points[:, 1] = np.tile(yz[:, 0], x_planes.size)
    points[:, 2] = np.tile(yz[:, 1], x_planes.size)

    layer_offsets = np.arange(x_planes.size - 1, dtype=np.int64) * nodes_per_plane
    lower = (
        local_triangles[None, :, :] + layer_offsets[:, None, None]
    ).reshape(-1, 3)
    upper = lower + nodes_per_plane
    connectivity = np.column_stack(
        (np.full(lower.shape[0], 6, dtype=np.int64), lower, upper)
    ).ravel()
    cell_types = np.full(lower.shape[0], pv.CellType.WEDGE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(connectivity, cell_types, points)
    grid.cell_data["active_grid"] = np.zeros(grid.n_cells, dtype=np.uint8)
    return grid


def transformed_surface(grid: pv.DataSet, vertical_exaggeration: float) -> pv.PolyData:
    """Extract a clean exterior surface and display depth downward."""
    surface = grid.extract_surface(algorithm="dataset_surface").clean()
    points = np.asarray(surface.points).copy()
    points[:, 2] *= -vertical_exaggeration
    surface.points = points
    return surface


def cells_on_axis_plane(
    surface: pv.PolyData,
    axis: int,
    coordinate: float,
    tolerance: float,
) -> pv.DataSet | None:
    """Extract exterior surface cells lying on a coordinate boundary plane."""
    if axis not in (0, 1):
        raise ValueError("Side-boundary axis must be x (0) or y (1)")
    centers = np.asarray(surface.cell_centers().points)
    cell_ids = np.flatnonzero(
        np.isclose(centers[:, axis], coordinate, rtol=0.0, atol=tolerance)
    )
    if cell_ids.size == 0:
        return None
    return surface.extract_cells(cell_ids)


def split_cells_on_z_plane(
    surface: pv.DataSet,
    z_coordinate: float,
    tolerance: float,
) -> tuple[pv.DataSet, pv.DataSet | None]:
    """Split a surface into a constant-z boundary and all remaining cells."""
    centers = np.asarray(surface.cell_centers().points)
    on_plane = np.isclose(
        centers[:, 2], z_coordinate, rtol=0.0, atol=tolerance
    )
    if not np.any(on_plane):
        return surface, None
    boundary = surface.extract_cells(np.flatnonzero(on_plane))
    remainder = surface.extract_cells(np.flatnonzero(~on_plane))
    return remainder, boundary


def top_seal_cell_mask(dataset: pv.DataSet) -> np.ndarray:
    """Select the complete top-seal package, including its refined interbeds."""
    required = (STRATIGRAPHIC_UNIT, ROCK_REGION)
    missing = [name for name in required if name not in dataset.cell_data]
    if missing:
        raise KeyError(f"Missing top-seal classification arrays: {missing}")

    unit_ids = np.asarray(dataset.cell_data[STRATIGRAPHIC_UNIT], dtype=np.int32)
    rock_regions = np.asarray(dataset.cell_data[ROCK_REGION], dtype=np.int32)

    # Outside the refined fault neighborhood, rock region 2 is the regional
    # top seal. Inside it, positive unit IDs replace that body with the full
    # interbedded stratigraphy. Their union is the complete top-seal interval.
    return (rock_regions == 2) | (unit_ids > 0)


def assign_geologic_regions(
    surface: pv.PolyData,
    vertical_exaggeration: float,
) -> None:
    """Classify visible active cells as overburden, top seal, or reservoir."""
    if STRATIGRAPHIC_UNIT not in surface.cell_data:
        raise KeyError(f"{STRATIGRAPHIC_UNIT} is missing from the active surface")

    unit_ids = np.asarray(surface.cell_data[STRATIGRAPHIC_UNIT], dtype=np.int32)
    top_seal = top_seal_cell_mask(surface)
    if not np.any(top_seal):
        raise ValueError("No top-seal stratigraphic units are present")

    centers = np.asarray(surface.cell_centers().points)
    fault_normal = centers[:, 1]
    depth = -centers[:, 2] / vertical_exaggeration
    seal_units = np.unique(unit_ids[top_seal])
    median_depth = {
        int(unit): float(np.median(depth[unit_ids == unit]))
        for unit in seal_units
    }
    shallow_unit = min(median_depth, key=median_depth.get)
    deep_unit = max(median_depth, key=median_depth.get)

    def unit_depth_curve(unit: int) -> tuple[np.ndarray, np.ndarray]:
        mask = unit_ids == unit
        keys = np.round(fault_normal[mask], decimals=2)
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        depth_sum = np.zeros(unique_keys.size, dtype=np.float64)
        counts = np.zeros(unique_keys.size, dtype=np.int64)
        np.add.at(depth_sum, inverse, depth[mask])
        np.add.at(counts, inverse, 1)
        return unique_keys, depth_sum / counts

    shallow_y, shallow_depth = unit_depth_curve(shallow_unit)
    deep_y, deep_depth = unit_depth_curve(deep_unit)
    shallow_at_cell = np.interp(fault_normal, shallow_y, shallow_depth)
    deep_at_cell = np.interp(fault_normal, deep_y, deep_depth)
    seal_midpoint = 0.5 * (shallow_at_cell + deep_at_cell)

    # Region IDs: 1 = overburden, 2 = top seal, 3 = storage reservoir.
    regions = np.ones(surface.n_cells, dtype=np.uint8)
    regions[top_seal] = 2
    regions[(~top_seal) & (depth >= seal_midpoint)] = 3
    surface.cell_data[GEOLOGIC_REGION] = regions


def build_full_fault_surface(
    fault_node_coordinates: Path,
    x_planes: np.ndarray,
    vertical_exaggeration: float,
) -> pv.StructuredGrid:
    """Extrude the exact source-node listric fault curve along strike."""
    if not fault_node_coordinates.is_file():
        raise FileNotFoundError(fault_node_coordinates)
    source = loadmat(fault_node_coordinates)
    if "fnodcoord" not in source:
        raise KeyError(f"fnodcoord is missing from {fault_node_coordinates}")
    coordinates = np.asarray(source["fnodcoord"], dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("fnodcoord must be an N-by-3 coordinate array")

    curve = coordinates[np.argsort(coordinates[:, 2])]
    if np.any(np.diff(curve[:, 2]) < 0):
        raise ValueError("The sorted fault curve is not depth-monotone")
    x_grid, _ = np.meshgrid(
        x_planes, np.arange(curve.shape[0]), indexing="ij"
    )
    y_grid = np.broadcast_to(curve[:, 1], x_grid.shape)
    depth_grid = np.broadcast_to(curve[:, 2], x_grid.shape)
    z_grid = -vertical_exaggeration * depth_grid
    return pv.StructuredGrid(x_grid, y_grid, z_grid)


def build_trace_fault_surface(
    trace_path: Path,
    x_planes: np.ndarray,
    vertical_exaggeration: float,
    sample_count: int = 180,
) -> pv.StructuredGrid:
    """Extrude a smooth y-depth trace supplied by structural anchor points."""
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    anchors = np.loadtxt(trace_path, delimiter=",", skiprows=1)
    anchors = np.atleast_2d(np.asarray(anchors, dtype=np.float64))
    if anchors.shape[1] != 2 or anchors.shape[0] < 2:
        raise ValueError(
            "The secondary-fault trace must contain at least two y_m,depth_m rows"
        )
    if not np.all(np.isfinite(anchors)):
        raise ValueError("The secondary-fault trace contains non-finite values")

    order = np.argsort(anchors[:, 1])
    y_anchor = anchors[order, 0]
    depth_anchor = anchors[order, 1]
    if np.any(np.diff(depth_anchor) <= 0):
        raise ValueError("Secondary-fault anchor depths must be strictly increasing")

    depth = np.linspace(depth_anchor[0], depth_anchor[-1], sample_count)
    y = PchipInterpolator(depth_anchor, y_anchor)(depth)
    x_grid, _ = np.meshgrid(x_planes, depth, indexing="ij")
    y_grid = np.broadcast_to(y, x_grid.shape)
    z_grid = np.broadcast_to(-vertical_exaggeration * depth, x_grid.shape)
    return pv.StructuredGrid(x_grid, y_grid, z_grid)


def crop_transparent_image(path: Path, padding: int = 32) -> None:
    """Remove the white VTK background and crop transparent margins."""
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA")).copy()
        rgb = rgba[:, :, :3].astype(np.float64)
        distance_from_white = 255.0 - np.min(rgb, axis=2)
        rgba[:, :, 3] = np.clip(5.0 * distance_from_white, 0.0, 255.0).astype(
            np.uint8
        )
        image = Image.fromarray(rgba)
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            raise RuntimeError("VTK rendered a fully transparent image")
        left, top, right, bottom = bbox
        image.crop(
            (
                max(0, left - padding),
                max(0, top - padding),
                min(image.width, right + padding),
                min(image.height, bottom + padding),
            )
        ).save(path, dpi=(600, 600), optimize=True)


def render(args: argparse.Namespace) -> None:
    """Render active stratigraphy, fault regions, and inactive context."""
    if not args.active_vtu.is_file():
        raise FileNotFoundError(args.active_vtu)
    for filename in ("nodes_coordinates.dat", "t.mat", "ucids_sc2_2D.mat"):
        if not (args.raw_grid / filename).is_file():
            raise FileNotFoundError(args.raw_grid / filename)
    if args.vertical_exaggeration <= 0:
        raise ValueError("vertical exaggeration must be positive")
    opacity_options = {
        "active opacity": args.active_opacity,
        "inactive opacity": args.inactive_opacity,
        "top-surface opacity": args.top_surface_opacity,
        "overburden opacity": args.overburden_opacity,
        "top-seal opacity": args.top_seal_opacity,
        "storage-reservoir opacity": args.storage_reservoir_opacity,
        "fault-region opacity": args.fault_region_opacity,
        "full-fault opacity": args.full_fault_opacity,
        "secondary-fault opacity": args.secondary_fault_opacity,
    }
    for option_name, value in opacity_options.items():
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{option_name} must be between 0 and 1")
    if args.injector_point_size <= 0:
        raise ValueError("injector point size must be positive")
    if args.injector is not None:
        injector_values = np.asarray(args.injector, dtype=np.float64)
        if not np.all(np.isfinite(injector_values)):
            raise ValueError("injector coordinates must be finite")
        if injector_values[2] < 0:
            raise ValueError("injector depth must be positive downward")

    active_full = pv.read(args.active_vtu)
    source_active_cell_count = active_full.n_cells
    if (
        ROCK_REGION not in active_full.cell_data
        or FAULT_REGION not in active_full.cell_data
    ):
        raise KeyError("Required rock or fault region arrays are missing")
    x_planes = np.unique(np.asarray(active_full.points)[:, 0])
    if x_planes.size != 88:
        raise ValueError(f"Expected 88 layer planes, found {x_planes.size}")

    # Keep the complete fault surfaces even when the reservoir body is cut
    # away, so the internal fault geometry remains directly visible.
    predict_fault = active_full.threshold(
        value=(0.75, 1.25), scalars=FAULT_REGION, preference="cell"
    )
    other_fault = active_full.threshold(
        value=(1.75, 2.25), scalars=FAULT_REGION, preference="cell"
    )

    active = active_full
    inactive = build_inactive_grid(args.raw_grid, x_planes)
    source_inactive_cell_count = inactive.n_cells
    if args.cutaway_y > 0:
        active_centers = active.cell_centers().points
        inactive_centers = inactive.cell_centers().points
        active = active.extract_cells(active_centers[:, 1] >= args.cutaway_y)
        inactive = inactive.extract_cells(inactive_centers[:, 1] >= args.cutaway_y)
    inactive_surface = transformed_surface(inactive, args.vertical_exaggeration)
    active_surface = transformed_surface(active, args.vertical_exaggeration)
    assign_geologic_regions(active_surface, args.vertical_exaggeration)
    top_seal_cells = active.extract_cells(top_seal_cell_mask(active))
    overburden_surface = active_surface.threshold(
        value=(0.75, 1.25), scalars=GEOLOGIC_REGION, preference="cell"
    )
    top_boundary_surface = None
    if args.top_surface_opacity is not None:
        top_z = overburden_surface.bounds[5]
        z_span = top_z - overburden_surface.bounds[4]
        z_tolerance = max(1.0e-6, 1.0e-8 * z_span)
        overburden_surface, top_boundary_surface = split_cells_on_z_plane(
            overburden_surface, top_z, z_tolerance
        )
    # Extracting the top-seal cells separately exposes the package interfaces;
    # those interfaces are internal and therefore absent from active_surface.
    top_seal_surface = transformed_surface(
        top_seal_cells, args.vertical_exaggeration
    )
    if args.opaque_active_sides and STRATIGRAPHY_REGION not in top_seal_surface.cell_data:
        raise KeyError(
            f"{STRATIGRAPHY_REGION} is required to delineate interbeds on "
            "the active-domain side surfaces"
        )
    storage_reservoir_surface = active_surface.threshold(
        value=(2.75, 3.25), scalars=GEOLOGIC_REGION, preference="cell"
    )

    predict_fault_surface = transformed_surface(
        predict_fault, args.vertical_exaggeration
    )
    other_fault_surface = transformed_surface(other_fault, args.vertical_exaggeration)
    full_fault_surface = None
    if args.delineate_full_fault:
        full_fault_surface = build_full_fault_surface(
            args.fault_node_coordinates,
            x_planes,
            vertical_exaggeration=args.vertical_exaggeration,
        )
    secondary_fault_surface = None
    if args.secondary_fault_trace is not None:
        secondary_fault_surface = build_trace_fault_surface(
            args.secondary_fault_trace,
            x_planes,
            vertical_exaggeration=args.vertical_exaggeration,
        )

    top_seal_side_regions: tuple[tuple[pv.DataSet, str], ...] = (
        (top_seal_surface, TOP_SEAL_COLOR),
    )
    if STRATIGRAPHY_REGION in top_seal_surface.cell_data:
        complete_seal_surface = top_seal_surface.threshold(
            value=(-0.25, 0.25),
            scalars=STRATIGRAPHY_REGION,
            preference="cell",
        )
        sand_interbed_surface = top_seal_surface.threshold(
            value=(0.75, 1.25),
            scalars=STRATIGRAPHY_REGION,
            preference="cell",
        )
        clay_interbed_surface = top_seal_surface.threshold(
            value=(1.75, 2.25),
            scalars=STRATIGRAPHY_REGION,
            preference="cell",
        )
        top_seal_side_regions = (
            (complete_seal_surface, TOP_SEAL_COLOR),
            (sand_interbed_surface, SAND_INTERBED_COLOR),
            (clay_interbed_surface, CLAY_INTERBED_COLOR),
        )
    active_region_surfaces = (
        (overburden_surface, OVERBURDEN_COLOR),
        *top_seal_side_regions,
        (storage_reservoir_surface, STORAGE_RESERVOIR_COLOR),
    )
    opaque_front_surfaces: list[tuple[pv.DataSet, str]] = []
    if args.opaque_front_xz:
        front_y = min(active_surface.bounds[2], inactive_surface.bounds[2])
        y_span = max(active_surface.bounds[3], inactive_surface.bounds[3]) - front_y
        y_tolerance = max(1.0e-6, 1.0e-8 * y_span)
        front_regions = [(inactive_surface, UNDERBURDEN_COLOR)]
        if not args.opaque_active_sides:
            front_regions.extend(active_region_surfaces)
        for region_surface, region_color in front_regions:
            front_surface = cells_on_axis_plane(
                region_surface, 1, front_y, y_tolerance
            )
            if front_surface is not None:
                opaque_front_surfaces.append((front_surface, region_color))

    opaque_active_side_surfaces: list[tuple[pv.DataSet, str]] = []
    interbed_side_outlines: list[pv.DataSet] = []
    if args.opaque_active_sides:
        x_min, x_max, y_min, y_max = active_surface.bounds[:4]
        x_tolerance = max(1.0e-6, 1.0e-8 * (x_max - x_min))
        y_tolerance = max(1.0e-6, 1.0e-8 * (y_max - y_min))
        side_planes = (
            (0, x_min, x_tolerance),
            (0, x_max, x_tolerance),
            (1, y_min, y_tolerance),
            (1, y_max, y_tolerance),
        )
        for region_surface, region_color in active_region_surfaces:
            for axis, coordinate, tolerance in side_planes:
                side_surface = cells_on_axis_plane(
                    region_surface, axis, coordinate, tolerance
                )
                if side_surface is not None:
                    opaque_active_side_surfaces.append(
                        (side_surface, region_color)
                    )
                    if axis == 0 and region_color in {
                        SAND_INTERBED_COLOR,
                        CLAY_INTERBED_COLOR,
                    }:
                        outline = side_surface.extract_surface(
                            algorithm="dataset_surface"
                        ).extract_feature_edges(
                            boundary_edges=True,
                            feature_edges=False,
                            manifold_edges=False,
                            non_manifold_edges=False,
                        )
                        if outline.n_cells:
                            interbed_side_outlines.append(outline)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.width, args.height),
        lighting="three lights",
    )
    plotter.set_background("white", top="white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)

    overburden_opacity = (
        min(1.0, 1.10 * args.active_opacity)
        if args.overburden_opacity is None
        else args.overburden_opacity
    )
    top_seal_opacity = (
        min(1.0, 2.20 * args.active_opacity)
        if args.top_seal_opacity is None
        else args.top_seal_opacity
    )
    storage_reservoir_opacity = (
        min(1.0, 1.15 * args.active_opacity)
        if args.storage_reservoir_opacity is None
        else args.storage_reservoir_opacity
    )

    plotter.add_mesh(
        inactive_surface,
        color=UNDERBURDEN_COLOR,
        opacity=args.inactive_opacity,
        smooth_shading=False,
        show_edges=True,
        edge_color="#74818D",
        line_width=0.20,
        ambient=0.55,
        diffuse=0.45,
        label="Underburden",
    )
    plotter.add_mesh(
        overburden_surface,
        color=OVERBURDEN_COLOR,
        smooth_shading=False,
        opacity=overburden_opacity,
        ambient=0.35,
        diffuse=0.72,
        specular=0.08,
        label="Overburden",
    )
    if top_boundary_surface is not None:
        plotter.add_mesh(
            top_boundary_surface,
            color=OVERBURDEN_COLOR,
            smooth_shading=False,
            opacity=args.top_surface_opacity,
            ambient=0.35,
            diffuse=0.72,
            specular=0.08,
        )
    plotter.add_mesh(
        top_seal_surface,
        color=TOP_SEAL_COLOR,
        smooth_shading=False,
        opacity=top_seal_opacity,
        ambient=0.35,
        diffuse=0.72,
        specular=0.08,
        label="Top seal",
    )
    plotter.add_mesh(
        storage_reservoir_surface,
        color=STORAGE_RESERVOIR_COLOR,
        smooth_shading=False,
        opacity=storage_reservoir_opacity,
        ambient=0.35,
        diffuse=0.72,
        specular=0.08,
        label="Storage reservoir",
    )
    # Make the complete near-side x-z boundary opaque. Add it before the fault
    # actors so transparent structural surfaces retain their normal
    # depth-peeling behavior everywhere else in the model.
    for front_surface, front_color in opaque_front_surfaces:
        plotter.add_mesh(
            front_surface,
            color=front_color,
            opacity=1.0,
            smooth_shading=False,
            ambient=0.38,
            diffuse=0.70,
            specular=0.05,
        )
    for side_surface, side_color in opaque_active_side_surfaces:
        plotter.add_mesh(
            side_surface,
            color=side_color,
            opacity=1.0,
            smooth_shading=False,
            ambient=0.38,
            diffuse=0.70,
            specular=0.05,
        )
    for outline in interbed_side_outlines:
        plotter.add_mesh(
            outline,
            color="#59362F",
            opacity=0.92,
            line_width=1.15,
            render_lines_as_tubes=True,
        )
    plotter.add_mesh(
        other_fault_surface,
        color=OTHER_FAULT_COLOR,
        opacity=args.fault_region_opacity,
        smooth_shading=True,
        ambient=0.42,
        diffuse=0.65,
        label="Fault",
    )
    plotter.add_mesh(
        predict_fault_surface,
        color=PREDICT_FAULT_COLOR,
        opacity=args.fault_region_opacity,
        smooth_shading=True,
        ambient=0.38,
        diffuse=0.72,
        label="Top-seal fault region",
    )
    if full_fault_surface is not None:
        plotter.add_mesh(
            full_fault_surface,
            color=OTHER_FAULT_COLOR,
            opacity=args.full_fault_opacity,
            smooth_shading=True,
            ambient=0.45,
            diffuse=0.62,
            label="Complete main fault",
        )
        fault_outline = full_fault_surface.extract_surface(
            algorithm="dataset_surface"
        ).extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(
            fault_outline,
            color=OTHER_FAULT_COLOR,
            line_width=3.0,
            render_lines_as_tubes=True,
        )
    if secondary_fault_surface is not None:
        plotter.add_mesh(
            secondary_fault_surface,
            color=SECONDARY_FAULT_COLOR,
            opacity=args.secondary_fault_opacity,
            smooth_shading=True,
            ambient=0.48,
            diffuse=0.58,
            label="Secondary structural fault",
        )
        secondary_outline = secondary_fault_surface.extract_surface(
            algorithm="dataset_surface"
        ).extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        plotter.add_mesh(
            secondary_outline,
            color=SECONDARY_FAULT_COLOR,
            line_width=3.0,
            render_lines_as_tubes=True,
        )

    if args.injector is not None:
        injector_x, injector_y, injector_depth = args.injector
        injector_center = (
            injector_x,
            injector_y,
            -args.vertical_exaggeration * injector_depth,
        )
        injector_point = np.asarray([injector_center], dtype=np.float64)
        # An always-visible screen-space point remains registered to the true
        # 3-D well centroid when the camera or model extent changes.
        plotter.add_point_labels(
            injector_point,
            [""],
            show_points=True,
            point_color=INJECTOR_COLOR,
            point_size=args.injector_point_size,
            render_points_as_spheres=True,
            always_visible=True,
            shape=None,
            bold=False,
            font_size=1,
            name="injector_marker",
        )
        if args.show_injector_label:
            plotter.add_point_labels(
                injector_point,
                ["Injector"],
                text_color="black",
                show_points=False,
                always_visible=True,
                shape=None,
                bold=False,
                font_size=18,
                name="injector_overlay",
            )

    bounds = np.asarray(inactive_surface.bounds)
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
    camera_distance = float(np.linalg.norm(extent))
    plotter.camera_position = [
        center
        + camera_distance
        * np.asarray(
            [
                args.camera_along_strike,
                args.camera_fault_normal,
                args.camera_elevation,
            ]
        ),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.10)
    if args.show_legend:
        plotter.add_legend(
            bcolor=None,
            border=False,
            face="rectangle",
            size=(0.20, 0.14),
            loc="upper right",
        )

    plotter.show(auto_close=False)
    try:
        plotter.screenshot(
            str(args.output), transparent_background=True, return_img=False
        )
    finally:
        plotter.close()
    crop_transparent_image(args.output)

    print(f"Source active cells: {source_active_cell_count}")
    print(f"Source inactive cells: {source_inactive_cell_count}")
    print(f"Rendered active cutaway cells: {active.n_cells}")
    print(f"Rendered inactive cutaway cells: {inactive.n_cells}")
    print(f"Layer planes: {x_planes.size}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    render(parse_args())
