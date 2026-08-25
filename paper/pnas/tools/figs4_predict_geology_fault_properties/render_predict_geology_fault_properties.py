#!/usr/bin/env python3
"""Compose an actual-geometry geology-to-PREDICT-property example.

The figure uses one internally consistent realization throughout:

* Step62 scenario 05 (medium sand proportion, nonuniform spacing),
* throw window W3,
* a source-verified PREDICT replay selected by the caller.

Panels (a) and (b) are vector Matplotlib geometry.  The four fine-scale
fault-core views are high-resolution, flat-lit PyVista renders embedded in a
vector PDF, so labels, axes, and color bars remain vector objects.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch, Polygon, Rectangle
from PIL import Image


HERE = Path(__file__).resolve().parent
PREDICT_ROOT = HERE.parents[3]
PNAS_TOOLS = HERE.parent


def default_grid_path() -> Path:
    """Locate the protected Step62 2-D VTU in a sibling grid checkout."""
    relative = Path(
        "setup_shaowen_resolution/grid_candidates/"
        "step_62_matched_upper_lower_transition/paraview/"
        "gom_step62_matched_upper_lower_transition_active_2d.vtu"
    )
    candidates = [
        PREDICT_ROOT.parent / "mrst_predict_sim_grid_integration" / relative,
        PREDICT_ROOT.parent / "mrst_predict_sim_grid_dev" / relative,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_GRID = default_grid_path()
DEFAULT_SCENARIOS = PREDICT_ROOT / "examples" / "thickness_scenario_designs.csv"
DEFAULT_RATIOS = (
    PREDICT_ROOT / "examples" / "footwall_sand_ratio_by_thickness_scenario.csv"
)
DEFAULT_REPLAY = HERE / "data" / "predict_w3_case01_medoid12_replay_compact.mat"

SAND = "#D8B365"
CLAY = "#8C6D5A"
OUTLINE = "#252525"
FAULT_FILL = "#000000"
WINDOW_FAULT_UNIT_IDS = {f"W{index}": index + 1 for index in range(1, 7)}
WINDOW_EXPECTED_CELL_COUNTS = {
    "W1": 62,
    "W2": 63,
    "W3": 62,
    "W4": 64,
    "W5": 62,
    "W6": 57,
}
VIEW_Y_KM = (9.75, 15.30)
VIEW_Z_KM = (1.32, 2.02)
VERTICAL_EXAGGERATION_A = 4.0
NORMAL_EXAGGERATION_C = 22.0
FAULT_DIP_DEG = 43.8
PANEL_C_Z_ROTATION_DEG = -60.0
# Keep the panel labels on their established common vertical guide.  Place the
# local-coordinate triad farther right, immediately below the lower-left corner
# of the first panel-(c) realization, without changing its accepted height.
PANEL_LABEL_X_AXES = 0.80
TRIAD_ORIGIN_AXES = np.array([3.00, -0.060])
MD_IN_M2 = 9.869233e-16


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "font.size": 11.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 11.0,
            "axes.linewidth": 0.72,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def import_helpers():
    for path in (str(PNAS_TOOLS), str(HERE)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import plot_fig2_real_stratigraphy as geology
    import plot_fig2_thickness_scenarios as scenarios
    import render_predict_fine_properties as properties
    import render_predict_smear_placement as materials

    return geology, scenarios, properties, materials


def material_raw_render(
    grid: pv.StructuredGrid,
    selected: pv.DataSet,
    color: str,
    output: Path,
    camera,
) -> None:
    """Render one categorical fault-core view with the common property camera."""
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1900))
    plotter.set_background("white")
    shell = grid.extract_surface(algorithm="dataset_surface")
    plotter.add_mesh(
        shell,
        color="#D8D8D8",
        opacity=0.050,
        show_edges=False,
        lighting=False,
    )
    plotter.add_mesh(
        selected.extract_surface(algorithm="dataset_surface"),
        color=color,
        opacity=1.0,
        show_edges=True,
        edge_color="#3A3735",
        line_width=0.45,
        lighting=False,
    )
    outline = shell.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(outline, color=OUTLINE, line_width=2.35, lighting=False)
    plotter.camera_position = camera
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.76)
    plotter.show(screenshot=str(output), auto_close=True)


def camera_for_right_dipping_material(grid: pv.StructuredGrid):
    """Keep the core vertical and view its internal architecture dipping right."""
    center = np.array(grid.center)
    length = grid.length
    position = center + np.array([0.72, 1.15, 0.52]) * length
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


def local_axes_from_transformed_grid(grid: pv.StructuredGrid) -> dict[str, np.ndarray]:
    """Recover the displayed +x, +y, and positive-down-dip +z directions.

    The directions are measured from the transformed StructuredGrid itself,
    rather than reconstructed from the requested rotation angle.  This keeps
    the vector triad synchronized with the actual rendered core if either the
    grid transform or camera is changed later.
    """
    nx, ny, nz = (int(value) for value in grid.dimensions)
    if min(nx, ny, nz) < 2:
        raise ValueError(f"Panel-(c) grid must be three-dimensional: {grid.dimensions}")
    origin = np.asarray(grid.points[0], dtype=float)
    step_indices = {"x": 1, "y": nx, "z": nx * ny}
    directions = {}
    for label, index in step_indices.items():
        direction = np.asarray(grid.points[index], dtype=float) - origin
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Invalid transformed panel-(c) {label}-axis vector")
        directions[label] = direction / norm
    return directions


def rotate_fault_core_about_z(grid: pv.StructuredGrid) -> pv.StructuredGrid:
    """Rotate the vertical core clockwise about its local z-axis."""
    grid.rotate_z(
        PANEL_C_Z_ROTATION_DEG,
        point=grid.center,
        inplace=True,
    )
    return grid


def render_fault_core_assets(
    replay: Path, output_dir: Path, helpers, asset_tag: str
) -> dict:
    _, _, properties, materials = helpers
    material_grid, material_summary = materials.load_grid(
        replay, NORMAL_EXAGGERATION_C
    )
    rotate_fault_core_about_z(material_grid)
    camera = camera_for_right_dipping_material(material_grid)
    paths = {
        "sand": output_dir / f"{asset_tag}_sand_raw.png",
        "clay": output_dir / f"{asset_tag}_clay_smear_raw.png",
        "kzz": output_dir / f"{asset_tag}_kzz_raw.png",
        "poro": output_dir / f"{asset_tag}_porosity_raw.png",
    }
    material_raw_render(
        material_grid,
        materials.subset(material_grid, clay=False),
        SAND,
        paths["sand"],
        camera,
    )
    material_raw_render(
        material_grid,
        materials.subset(material_grid, clay=True),
        CLAY,
        paths["clay"],
        camera,
    )

    property_grid, property_summary = properties.load_grid(
        replay, NORMAL_EXAGGERATION_C, "kzz"
    )
    rotate_fault_core_about_z(property_grid)
    k_clim = tuple(property_summary["log10_permeability_md_range"])
    phi_clim = tuple(property_summary["porosity_range"])
    property_camera = camera_for_right_dipping_material(property_grid)
    properties.render_panel(
        property_grid,
        "log10_permeability_md",
        "magma",
        k_clim,
        paths["kzz"],
        property_camera,
        False,
    )
    properties.render_panel(
        property_grid,
        "porosity",
        "viridis",
        phi_clim,
        paths["poro"],
        property_camera,
        False,
    )
    return {
        "paths": paths,
        "material_summary": material_summary,
        "property_summary": property_summary,
        "color_limits": {"kzz": k_clim, "poro": phi_clim},
        "panel_c_world_axes": local_axes_from_transformed_grid(property_grid),
        "panel_c_camera": property_camera,
    }


def crop_rgba(path: Path, pad: int = 20) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    array = np.asarray(image)
    mask = (array[:, :, 3] > 0) & np.any(array[:, :, :3] < 250, axis=2)
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return image
    return image.crop(
        (
            max(0, int(xs.min()) - pad),
            max(0, int(ys.min()) - pad),
            min(image.width, int(xs.max()) + pad + 1),
            min(image.height, int(ys.max()) + pad + 1),
        )
    )


def find_window_interface_segments(
    points,
    triangles,
    fault_unit_ids,
    a_layer_ids,
    side_codes,
    region_names,
):
    """Return PREDICT-consistent interfaces bracketing and separating W1-W6."""
    edge_cells = {}
    vertex_cells = {}
    for cell_index, triangle in enumerate(triangles):
        for vertex in triangle:
            vertex_cells.setdefault(int(vertex), []).append(cell_index)
        triangle = triangles[cell_index]
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_cells.setdefault(edge, []).append(cell_index)

    # Step62 unit interfaces correctly represent W1/W2 through W6/MM-UM.
    # The lower Step62 f_LM2/f_amp1 edge is intentionally excluded because
    # PREDICT's W1 hanging wall starts at A2, not at the base of A1.
    expected_pairs = {(unit_id, unit_id + 1) for unit_id in range(2, 8)}
    segments = [
        points[np.asarray(edge, dtype=int)]
        for edge, cells in edge_cells.items()
        if len(cells) == 2
        and tuple(sorted({int(fault_unit_ids[cell]) for cell in cells}))
        in expected_pairs
    ]
    adjacent_pairs = {
        tuple(sorted({int(fault_unit_ids[cell]) for cell in cells}))
        for cells in edge_cells.values()
        if len(cells) == 2
        and tuple(sorted({int(fault_unit_ids[cell]) for cell in cells}))
        in expected_pairs
    }
    if adjacent_pairs != expected_pairs:
        raise ValueError(
            "Unexpected PREDICT window adjacency: "
            f"expected {sorted(expected_pairs)}, found {sorted(adjacent_pairs)}"
        )

    # PREDICT W1 uses hanging-wall A2-A5 (SSSC, deepest to shallowest).
    # Locate the unique A1/A2 contact vertex on the hanging-wall fault edge.
    hangingwall_candidates = []
    for vertex, cells in vertex_cells.items():
        signatures = {
            (
                int(fault_unit_ids[cell]),
                int(side_codes[cell]),
                int(a_layer_ids[cell]),
            )
            for cell in cells
        }
        if (
            any(signature[0] == 2 for signature in signatures)
            and (0, 2, 1) in signatures
            and (0, 2, 2) in signatures
        ):
            hangingwall_candidates.append(vertex)
    if len(hangingwall_candidates) != 1:
        raise ValueError(
            "Expected one W1 hanging-wall A1/A2 contact vertex, found "
            f"{len(hangingwall_candidates)}"
        )
    hangingwall_vertex = hangingwall_candidates[0]
    hangingwall_point = np.asarray(points[hangingwall_vertex], dtype=float)

    # PREDICT constructs the top and bottom of each fault section at constant
    # depth. Intersect the A1/A2 contact depth with the opposite W1 edge
    # adjacent to LM2; do not project along the fault-normal direction.
    opposite_candidates = []
    for edge, cells in edge_cells.items():
        if len(cells) != 2:
            continue
        units = [int(fault_unit_ids[cell]) for cell in cells]
        if 2 not in units or 0 not in units:
            continue
        host_cell = cells[units.index(0)]
        if str(region_names[host_cell]) != "res_LM2":
            continue
        first, second = np.asarray(points[np.asarray(edge, dtype=int)], dtype=float)
        target_depth = float(hangingwall_point[1])
        depth_min, depth_max = sorted((float(first[1]), float(second[1])))
        if not depth_min - 1.0e-12 <= target_depth <= depth_max + 1.0e-12:
            continue
        depth_change = float(second[1] - first[1])
        if abs(depth_change) < 1.0e-12:
            continue
        fraction = (target_depth - float(first[1])) / depth_change
        footwall_y = float(first[0] + fraction * (second[0] - first[0]))
        intersection = np.asarray((footwall_y, target_depth), dtype=float)
        horizontal_length = abs(float(hangingwall_point[0] - footwall_y))
        opposite_candidates.append((horizontal_length, intersection))
    if not opposite_candidates:
        raise ValueError("Could not locate the opposite W1 edge adjacent to LM2")
    distance, footwall_point = min(opposite_candidates, key=lambda item: item[0])
    if not 0.005 < distance < 0.080:
        raise ValueError(f"Unexpected corrected W1 boundary length: {distance} km")
    if not np.isclose(footwall_point[1], hangingwall_point[1], atol=1.0e-12):
        raise ValueError("Corrected W1 boundary is not horizontal")

    segments.append(np.vstack((footwall_point, hangingwall_point)))
    boundary_summary = {
        "basis": (
            "PREDICT constant-depth boundary through the hanging-wall A1/A2 "
            "contact; A1 excluded from W1"
        ),
        "footwall_yz_km": footwall_point.tolist(),
        "hangingwall_yz_km": hangingwall_point.tolist(),
        "length_m": 1000.0 * distance,
    }
    return segments, boundary_summary


def draw_geologic_scenario(ax, grid: Path, scenario_file: Path, ratio_file: Path, helpers):
    geology, scenarios_module, _, _ = helpers
    scenario_data = scenarios_module.read_scenarios(scenario_file)
    sand_percent = scenarios_module.read_sand_percent(ratio_file)[5]
    points, triangles, layer_ids, fault_mask, fault_segments = (
        geology.load_step62_geometry(grid)
    )
    material_by_layer = geology.scenario_material_by_layer(
        scenario_data, "scenario_05_medium_sand_nonuniform"
    )
    stratigraphy_mask = layer_ids > 0
    fault_polygons = geology.dissolve_triangles(points, triangles, fault_mask)
    step62 = pv.read(grid)
    fault_unit_ids = np.asarray(step62.cell_data["fault_unit_id"], dtype=int)
    a_layer_ids = np.asarray(step62.cell_data["a_layer_id"], dtype=int)
    side_codes = np.asarray(step62.cell_data["side_code"], dtype=int)
    region_names = np.asarray(step62.cell_data["region_name"])
    window_annotations = {}
    for label, fault_unit_id in WINDOW_FAULT_UNIT_IDS.items():
        window_mask = fault_unit_ids == fault_unit_id
        cell_count = int(window_mask.sum())
        expected_count = WINDOW_EXPECTED_CELL_COUNTS[label]
        if cell_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} Step62 cells in {label}, found {cell_count}"
            )
        center = np.asarray(points[triangles[window_mask]], dtype=float).mean(
            axis=(0, 1)
        )
        window_annotations[label] = {
            "fault_unit_id": fault_unit_id,
            "cell_count": cell_count,
            "center_y_km": float(center[0]),
            "center_z_km": float(center[1]),
        }
    window_interface_segments, w1_lower_boundary = find_window_interface_segments(
        points,
        triangles,
        fault_unit_ids,
        a_layer_ids,
        side_codes,
        region_names,
    )
    window_annotations["W1"]["lower_boundary"] = w1_lower_boundary

    ax.add_collection(
        PolyCollection(
            fault_polygons,
            facecolors=FAULT_FILL,
            edgecolors="none",
            linewidths=0,
            antialiaseds=False,
            zorder=1,
        )
    )
    for code, color in (("S", SAND), ("C", CLAY)):
        assigned_ids = [
            layer_id
            for layer_id, material in material_by_layer.items()
            if material == code
        ]
        polygons = geology.dissolve_triangles(
            points,
            triangles,
            stratigraphy_mask & np.isin(layer_ids, assigned_ids),
        )
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors=color,
                edgecolors="none",
                linewidths=0,
                antialiaseds=False,
                zorder=2,
            )
        )
    ax.add_collection(
        LineCollection(
            fault_segments,
            colors=OUTLINE,
            linewidths=0.68,
            capstyle="round",
            joinstyle="round",
            zorder=3,
        )
    )
    ax.add_collection(
        LineCollection(
            window_interface_segments,
            colors="white",
            linewidths=0.86,
            capstyle="round",
            joinstyle="round",
            zorder=4,
        )
    )
    # Place each label immediately beside its exact Step62 throw-window
    # centroid. No leaders, halos, or colored overlays are used.
    for label, annotation in window_annotations.items():
        ax.text(
            annotation["center_y_km"] + 0.080,
            annotation["center_z_km"],
            label,
            color=OUTLINE,
            fontsize=11.0,
            ha="left",
            va="center",
            zorder=5,
        )

    ax.set_xlim(*VIEW_Y_KM)
    ax.set_ylim(VIEW_Z_KM[1], VIEW_Z_KM[0])
    ax.set_aspect(VERTICAL_EXAGGERATION_A, adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_color(OUTLINE)
        spine.set_linewidth(0.75)
    ax.set_title("Geologic scenario", pad=5.5)
    ax.legend(
        handles=[
            Patch(facecolor=SAND, edgecolor=OUTLINE, linewidth=0.45, label="Sand interbed"),
            Patch(facecolor=CLAY, edgecolor=OUTLINE, linewidth=0.45, label="Clay interbed"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.012, -0.025),
        ncol=1,
        frameon=False,
        handlelength=1.2,
        handleheight=0.7,
        labelspacing=0.22,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    return float(sand_percent), window_annotations


def draw_throw_window(ax, fault_thickness: float) -> dict:
    """Draw W3 using the PREDICT stratigraphic and fault dips."""
    fault_dip = FAULT_DIP_DEG
    footwall_dip = 0.0
    hangingwall_dip = -9.7683
    apparent_layer_thickness = 35.8537
    # PREDICT's apparent layer thickness is measured along the fault.  These
    # are the exact layer/fault intersections used by FaultedSection.plotStrati.
    along_fault = np.arange(5, dtype=float) * apparent_layer_thickness
    elevation = along_fault * np.sin(np.deg2rad(fault_dip))
    left_fault_x = -along_fault * np.cos(np.deg2rad(fault_dip))
    right_fault_x = left_fault_x + fault_thickness
    height = float(elevation[-1])
    # Wider lateral context makes the listric-fault section read with the
    # same steep orientation as panel (a), without changing any input dip.
    # Start from the 35%-shortened design, then widen the visible W3 section
    # by 25% while retaining its established panel height and plotting frame.
    lateral_margin = 300.0 * 0.65 * 1.25
    left_bound = float(left_fault_x[-1] - lateral_margin)
    right_bound = float(fault_thickness + lateral_margin)
    # Retain the previous horizontal plotting scale so the 35% reduction is
    # visible instead of being stretched back across the full panel width.
    # Shift the schematic left within panel (b), reducing the horizontal gap
    # between its left boundary and the panel index without changing its size.
    panel_shift_m = 50.0
    display_left_bound = float(left_fault_x[-1] - 300.0 + panel_shift_m)
    display_right_bound = float(fault_thickness + 300.0 + panel_shift_m)

    # Scenario 05, W3: SCCC on the footwall and SSCC on the hanging wall,
    # ordered from the deep base upward.  Footwall contacts are horizontal;
    # hanging-wall contacts retain the true -9.7683-degree input dip.
    fw_materials = "SCCC"
    hw_materials = "SSCC"
    hw_outer_elevation = elevation + np.tan(np.deg2rad(hangingwall_dip)) * (
        right_fault_x - right_bound
    )

    def depth(z):
        return height - np.asarray(z, dtype=float)

    def material_runs(materials: str):
        """Yield contiguous lithology runs so same-material units have no seams."""
        start = 0
        for stop in range(1, len(materials) + 1):
            if stop == len(materials) or materials[stop] != materials[start]:
                yield start, stop, materials[start]
                start = stop

    # Draw only material boundaries in this schematic. The thinner source-unit
    # divisions remain part of the PREDICT input but are intentionally omitted.
    for start, stop, material in material_runs(fw_materials):
        polygon = np.array(
            [
                [left_bound, depth(elevation[start])],
                [left_fault_x[start], depth(elevation[start])],
                [left_fault_x[stop], depth(elevation[stop])],
                [left_bound, depth(elevation[stop])],
            ]
        )
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=SAND if material == "S" else CLAY,
                edgecolor="none",
                linewidth=0,
                antialiased=False,
            )
        )

    for start, stop, material in material_runs(hw_materials):
        polygon = np.array(
            [
                [right_fault_x[start], depth(elevation[start])],
                [right_bound, depth(hw_outer_elevation[start])],
                [right_bound, depth(hw_outer_elevation[stop])],
                [right_fault_x[stop], depth(elevation[stop])],
            ]
        )
        ax.add_patch(
            Polygon(
                polygon,
                closed=True,
                facecolor=SAND if material == "S" else CLAY,
                edgecolor="none",
                linewidth=0,
                antialiased=False,
            )
        )

    core_polygon = np.column_stack(
        (
            np.r_[left_fault_x, right_fault_x[::-1]],
            np.r_[depth(elevation), depth(elevation)[::-1]],
        )
    )
    ax.add_patch(
        Polygon(
            core_polygon,
            closed=True,
            facecolor="#ECEAE4",
            edgecolor=OUTLINE,
            linewidth=1.15,
            zorder=4,
        )
    )

    # Sparse in-panel labels preserve the lithologic interpretation without
    # hiding the actual inclined contacts.
    label_inset_m = 15.0
    ax.text(left_bound + label_inset_m, depth(0.5 * elevation[1]), "Sand", color=OUTLINE,
            ha="left", va="center", fontsize=11.0)
    ax.text(left_bound + label_inset_m, depth(0.5 * (elevation[3] + elevation[4])), "Clay",
            color="white", ha="left", va="center", fontsize=11.0)
    top_core_x = 0.5 * (left_fault_x[-1] + right_fault_x[-1])
    top_core_depth = float(depth(elevation[-1]))
    ax.annotate(
        "Fault core",
        xy=(top_core_x, top_core_depth),
        xytext=(top_core_x, top_core_depth - 18.0),
        ha="center",
        va="bottom",
        fontsize=11.0,
        color=OUTLINE,
        zorder=6,
        arrowprops={
            "arrowstyle": "-|>",
            "color": OUTLINE,
            "lw": 0.75,
            "mutation_scale": 7.0,
            "shrinkA": 2.5,
            "shrinkB": 2.0,
        },
    )
    all_depths = np.r_[depth(elevation), depth(hw_outer_elevation)]
    depth_pad = 0.025 * float(np.ptp(all_depths))
    ax.set_xlim(display_left_bound, display_right_bound)
    ax.set_ylim(float(all_depths.max() + depth_pad), float(all_depths.min() - depth_pad))
    ax.set_aspect("auto")
    ax.set_axis_off()
    schematic_center_x = 0.5 * (left_bound + right_bound)
    schematic_center_fraction = (
        (schematic_center_x - display_left_bound)
        / (display_right_bound - display_left_bound)
    )
    ax.set_title("Throw window W3", x=schematic_center_fraction, pad=5.5)
    return {
        "window": "W3",
        "fault_dip_deg": fault_dip,
        "footwall_dip_deg": footwall_dip,
        "hangingwall_dip_deg": hangingwall_dip,
        "vertical_throw_m": height,
        "fault_normal_thickness_m": fault_thickness,
    }


def add_local_triad(ax, world_directions: dict[str, np.ndarray], camera) -> None:
    """Draw equal-length fault-local axes parallel to the rendered box edges."""
    # Project directions measured from the transformed grid through the exact
    # orthographic camera used for the PyVista views.  The grid loader reverses
    # its display-z coordinate, so increasing grid k is positive down dip.
    position = np.asarray(camera[0], dtype=float)
    focal_point = np.asarray(camera[1], dtype=float)
    view_up = np.asarray(camera[2], dtype=float)
    view = focal_point - position
    view /= np.linalg.norm(view)
    right = np.cross(view, view_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, view)
    up /= np.linalg.norm(up)

    bbox = ax.get_position()
    width_in = bbox.width * ax.figure.get_figwidth()
    height_in = bbox.height * ax.figure.get_figheight()
    arrow_length_in = 0.19
    # The triad uses the dedicated narrow column, but is shifted toward the
    # first fault-core view so the intervening whitespace is minimal.  Drawing
    # remains unclipped to keep the panel label independently left-aligned.
    origin = TRIAD_ORIGIN_AXES.copy()
    for label, world_direction in world_directions.items():
        display_direction = np.array(
            [np.dot(world_direction, right), np.dot(world_direction, up)]
        )
        display_direction /= np.linalg.norm(display_direction)
        vector = arrow_length_in * np.array(
            [display_direction[0] / width_in, display_direction[1] / height_in]
        )
        tip = origin + vector
        ax.annotate(
            "",
            xy=tip,
            xytext=origin,
            xycoords=ax.transAxes,
            arrowprops={
                "arrowstyle": "-|>",
                "color": OUTLINE,
                "lw": 0.62,
                "mutation_scale": 4.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )
        label_gap_in = 0.032
        label_offset = label_gap_in * np.array(
            [display_direction[0] / width_in, display_direction[1] / height_in]
        )
        ax.text(
            *(tip + label_offset),
            rf"${label}$",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.0,
            clip_on=False,
        )


def compose(
    output_stem: Path,
    raw: dict,
    grid: Path,
    scenario_file: Path,
    ratio_file: Path,
    helpers,
) -> dict:
    configure_style()
    images = {key: crop_rgba(path) for key, path in raw["paths"].items()}
    fig = plt.figure(figsize=(7.25, 5.55))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=(0.37, 0.63),
        left=0.025,
        right=0.995,
        bottom=0.070,
        top=0.965,
        hspace=0.17,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=(2.05, 0.95), wspace=0.03)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    # Move panel (b) as one unit—including its index, title, annotations, and
    # schematic—rather than shifting only the data inside the axes.
    panel_b_box = ax_b.get_position()
    panel_b_group_shift = 0.025
    ax_b.set_position(
        [
            panel_b_box.x0 - panel_b_group_shift,
            panel_b_box.y0,
            panel_b_box.width,
            panel_b_box.height,
        ]
    )
    sand_percent, window_annotations = draw_geologic_scenario(
        ax_a, grid, scenario_file, ratio_file, helpers
    )
    physical_bounds = np.asarray(
        raw["property_summary"]["physical_bounds_m"], dtype=float
    )
    fault_thickness = float(physical_bounds[1, 0] - physical_bounds[0, 0])
    window_summary = draw_throw_window(ax_b, fault_thickness)
    ax_b.text(-0.16, 1.18, "(b)", transform=ax_b.transAxes, ha="left", va="top", fontsize=11.0)

    bottom = outer[1].subgridspec(
        2,
        5,
        height_ratios=(1.0, 0.060),
        width_ratios=(0.12, 1, 1, 1, 1),
        wspace=0.040,
        hspace=0.028,
    )
    ax_triad = fig.add_subplot(bottom[0, 0])
    axes_c = [fig.add_subplot(bottom[0, index]) for index in range(1, 5)]
    cbar_slots = [fig.add_subplot(bottom[1, index]) for index in range(1, 5)]
    titles = [
        "Sand",
        "Clay smear",
        "Permeability",
        "Porosity",
    ]
    keys = ["sand", "clay", "kzz", "poro"]
    for axis, key, title in zip(axes_c, keys, titles, strict=True):
        axis.imshow(images[key], interpolation="lanczos")
        axis.set_axis_off()
        axis.set_title(title, pad=2.2, fontsize=11.0)
    ax_triad.set_axis_off()
    # Use a fixed common vertical guide for the two left-hand panel indices;
    # the coordinate triad can then be positioned independently below Sand.
    triad_bbox = ax_triad.get_position()
    common_left_figure = (
        triad_bbox.x0 + PANEL_LABEL_X_AXES * triad_bbox.width
    )
    panel_a_bbox = ax_a.get_position()
    panel_a_label_x = (
        common_left_figure - panel_a_bbox.x0
    ) / panel_a_bbox.width
    ax_a.text(
        panel_a_label_x,
        1.18,
        "(a)",
        transform=ax_a.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        clip_on=False,
    )
    ax_triad.text(
        PANEL_LABEL_X_AXES,
        1.055,
        "(c)",
        transform=ax_triad.transAxes,
        ha="left",
        va="top",
        fontsize=11.0,
        clip_on=False,
    )
    add_local_triad(
        ax_triad,
        raw["panel_c_world_axes"],
        raw["panel_c_camera"],
    )

    for slot in cbar_slots:
        slot.set_axis_off()
    k_clim = tuple(raw["color_limits"]["kzz"])
    phi_clim = tuple(raw["color_limits"]["poro"])
    k_norm = mpl.colors.Normalize(vmin=k_clim[0], vmax=k_clim[1])
    phi_norm = mpl.colors.Normalize(vmin=phi_clim[0], vmax=phi_clim[1])

    def short_colorbar_axis(slot):
        bbox = slot.get_position()
        return fig.add_axes(
            [bbox.x0 + 0.19 * bbox.width, bbox.y0 + 0.24 * bbox.height,
             0.62 * bbox.width, 0.50 * bbox.height]
        )

    k_cax = short_colorbar_axis(cbar_slots[2])
    phi_cax = short_colorbar_axis(cbar_slots[3])
    cb_k = fig.colorbar(
        mpl.cm.ScalarMappable(norm=k_norm, cmap="magma"),
        cax=k_cax,
        orientation="horizontal",
        ticks=[k_clim[0], k_clim[1]],
    )
    cb_k.set_label(
        r"$\log_{10}\!\left(k_{zz}\,[\mathrm{mD}]\right)$",
        labelpad=1.0,
        fontsize=11.0,
    )
    cb_k.ax.set_xticklabels([f"{k_clim[0]:.2f}", f"{k_clim[1]:.2f}"])
    cb_phi = fig.colorbar(
        mpl.cm.ScalarMappable(norm=phi_norm, cmap="viridis"),
        cax=phi_cax,
        orientation="horizontal",
        ticks=[phi_clim[0], phi_clim[1]],
    )
    cb_phi.set_label(r"$\phi$ [-]", labelpad=1.0, fontsize=11.0)
    cb_phi.ax.set_xticklabels([f"{phi_clim[0]:.2f}", f"{phi_clim[1]:.2f}"])
    for axis in (k_cax, phi_cax):
        axis.tick_params(direction="in", length=2.2, width=0.55, pad=1.1, labelsize=11.0)
        axis.xaxis.set_label_position("bottom")

    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(pdf, dpi=600, bbox_inches="tight", pad_inches=0.025, facecolor="white")
    plt.close(fig)
    return {
        "png": str(png),
        "pdf": str(pdf),
        "sand_percent": sand_percent,
        "panel_a_window_annotations": window_annotations,
        "window": window_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--ratios", type=Path, default=DEFAULT_RATIOS)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PREDICT_ROOT
        / "paper"
        / "pnas"
        / "figures"
        / "fig4_predict_geology_fault_properties",
    )
    parser.add_argument("--reuse-raw", action="store_true")
    parser.add_argument("--faulting-depth-m", type=float, default=50.0)
    parser.add_argument("--sand-vcl", type=float, default=0.10)
    parser.add_argument("--clay-vcl", type=float, default=0.40)
    parser.add_argument("--sample-index", type=int, default=12)
    parser.add_argument("--seed", type=int, default=530101740)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def portable_path(path: Path) -> str:
    """Record repository and sibling-checkout paths without host-specific roots."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PREDICT_ROOT.resolve()).as_posix()
    except ValueError:
        pass
    try:
        relative = resolved.relative_to(PREDICT_ROOT.parent.resolve()).as_posix()
        return f"../{relative}"
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    for path in (args.grid, args.scenarios, args.ratios, args.replay):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    helpers = import_helpers()
    asset_tag = args.output_stem.name
    raw_dir = HERE / "_cache"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_paths = {
        "sand": raw_dir / f"{asset_tag}_sand_raw.png",
        "clay": raw_dir / f"{asset_tag}_clay_smear_raw.png",
        "kzz": raw_dir / f"{asset_tag}_kzz_raw.png",
        "poro": raw_dir / f"{asset_tag}_porosity_raw.png",
    }
    if args.reuse_raw and all(path.is_file() for path in raw_paths.values()):
        _, _, properties, materials = helpers
        _, material_summary = materials.load_grid(
            args.replay, NORMAL_EXAGGERATION_C
        )
        property_grid, property_summary = properties.load_grid(
            args.replay, NORMAL_EXAGGERATION_C, "kzz"
        )
        rotate_fault_core_about_z(property_grid)
        property_camera = camera_for_right_dipping_material(property_grid)
        raw = {
            "paths": raw_paths,
            "material_summary": material_summary,
            "property_summary": property_summary,
            "color_limits": {
                "kzz": tuple(property_summary["log10_permeability_md_range"]),
                "poro": tuple(property_summary["porosity_range"]),
            },
            "panel_c_world_axes": local_axes_from_transformed_grid(property_grid),
            "panel_c_camera": property_camera,
        }
    else:
        raw = render_fault_core_assets(
            args.replay, raw_dir, helpers, asset_tag
        )
    summary = compose(
        args.output_stem,
        raw,
        args.grid,
        args.scenarios,
        args.ratios,
        helpers,
    )
    summary["png"] = portable_path(Path(summary["png"]))
    summary["pdf"] = portable_path(Path(summary["pdf"]))
    summary.update(
        {
            "scenario": "scenario_05_medium_sand_nonuniform",
            "faulting_depth_m": args.faulting_depth_m,
            "sand_vcl": args.sand_vcl,
            "clay_vcl": args.clay_vcl,
            "sample_index": args.sample_index,
            "seed": args.seed,
            "fault_normal_exaggeration_panel_c": NORMAL_EXAGGERATION_C,
            "panel_a_vertical_exaggeration": VERTICAL_EXAGGERATION_A,
            "panel_c_core_orientation": "vertical",
            "panel_c_camera_offset": [0.72, 1.15, 0.52],
            "panel_c_rotation_about_z_deg": PANEL_C_Z_ROTATION_DEG,
            "panel_c_world_axes": raw["panel_c_world_axes"],
            "source_replay": portable_path(args.replay),
            "source_grid": portable_path(args.grid),
            "material_summary": raw["material_summary"],
            "property_summary": raw["property_summary"],
        }
    )
    metadata = args.output_stem.with_suffix(".json")
    serialized = json.dumps(summary, indent=2, default=json_default)
    metadata.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
