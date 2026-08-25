"""Render actual PREDICT sand-filled material and clay-smear placement."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from scipy.io import loadmat


SAND = "#D9A24A"
CLAY = "#6B3E32"
OUTLINE = "#2A2A2A"


def load_grid(path: Path, normal_exaggeration: float) -> tuple[pv.StructuredGrid, dict]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    coords = np.asarray(data["coords"], dtype=float)
    dims = np.asarray(data["cartDims"], dtype=int).ravel()
    is_smear = np.asarray(data["isSmear"], dtype=np.uint8).ravel(order="F")

    if coords.shape[1] != 3 or coords.shape[0] != int(np.prod(dims + 1)):
        raise ValueError("Unexpected replay-grid geometry.")
    if is_smear.size != int(np.prod(dims)):
        raise ValueError("Unexpected material-mask size.")

    points = coords.copy()
    points[:, 0] *= normal_exaggeration
    # PREDICT local z is positive down-dip; display it downward.
    points[:, 2] *= -1.0

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims + 1
    grid.cell_data["is_clay_smear"] = is_smear

    summary = {
        "dims": dims,
        "sand_fraction": float(np.mean(is_smear == 0)),
        "clay_fraction": float(np.mean(is_smear == 1)),
        "bounds_physical": np.vstack((coords.min(axis=0), coords.max(axis=0))),
    }
    return grid, summary


def subset(grid: pv.StructuredGrid, clay: bool) -> pv.DataSet:
    lo, hi = (0.5, 1.5) if clay else (-0.5, 0.5)
    return grid.threshold((lo, hi), scalars="is_clay_smear", preference="cell")


def camera_for(grid: pv.StructuredGrid):
    center = np.array(grid.center)
    length = grid.length
    position = center + np.array([0.72, -1.15, 0.52]) * length
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


def add_local_coordinate_system(plotter: pv.Plotter, grid: pv.StructuredGrid) -> None:
    """Add the PREDICT fault-local x-y-z frame in the rendered coordinates."""
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    span_y = ymax - ymin
    span_z = zmax - zmin
    length = 0.085 * min(span_y, span_z)
    origin = np.array(
        [xmin - 0.08 * length, ymin - 0.18 * span_y, zmin + 0.16 * span_z],
        dtype=float,
    )
    directions = {
        "x": np.array([1.0, 0.0, 0.0]),       # fault normal
        "y": np.array([0.0, 1.0, 0.0]),       # along strike
        "z": np.array([0.0, 0.0, -1.0]),      # positive down dip
    }
    tips = []
    labels = []
    for label, direction in directions.items():
        arrow = pv.Arrow(
            start=origin,
            direction=direction,
            tip_length=0.28,
            tip_radius=0.105,
            shaft_radius=0.028,
            scale=length,
        )
        plotter.add_mesh(arrow, color=OUTLINE, lighting=False)
        tips.append(origin + 1.10 * length * direction)
        labels.append(label)
    plotter.add_mesh(pv.Sphere(radius=0.030 * length, center=origin), color=OUTLINE, lighting=False)
    plotter.add_point_labels(
        np.asarray(tips),
        labels,
        font_size=20,
        text_color=OUTLINE,
        show_points=False,
        shape=None,
        always_visible=True,
    )


def render_panel(
    grid: pv.StructuredGrid,
    selected: pv.DataSet,
    color: str,
    output: Path,
    camera,
    show_coordinate_system: bool,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1650, 1800))
    plotter.set_background("white")
    shell = grid.extract_surface(algorithm="dataset_surface")
    plotter.add_mesh(
        shell,
        color="#D6D6D6",
        opacity=0.055,
        show_edges=False,
        lighting=False,
    )
    plotter.add_mesh(
        selected.extract_surface(algorithm="dataset_surface"),
        color=color,
        opacity=1.0,
        show_edges=True,
        edge_color="#3A302C",
        line_width=0.65,
        lighting=False,
    )
    plotter.add_mesh(
        shell.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        ),
        color=OUTLINE,
        line_width=2.2,
        lighting=False,
    )
    if show_coordinate_system:
        add_local_coordinate_system(plotter, grid)
    plotter.camera_position = camera
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.76)
    plotter.show(screenshot=str(output), auto_close=True)


def crop_rgba(image: Image.Image, pad: int = 18) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba)
    mask = (arr[:, :, 3] > 0) & np.any(arr[:, :, :3] < 250, axis=2)
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return rgba
    box = (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(rgba.width, int(xs.max()) + pad + 1),
        min(rgba.height, int(ys.max()) + pad + 1),
    )
    return rgba.crop(box)


def compose(sand_path: Path, clay_path: Path, output_stem: Path, summary: dict) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )
    images = [crop_rgba(Image.open(sand_path)), crop_rgba(Image.open(clay_path))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.35), constrained_layout=True)
    labels = ["Sand", "Clay smear"]
    letters = ["(a)", "(b)"]
    for ax, image, label, letter in zip(axes, images, labels, letters):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(0.015, 0.985, letter, transform=ax.transAxes, ha="left", va="top")
        ax.set_title(label, pad=3.0, fontsize=11)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--normal-exaggeration", type=float, default=22.0)
    args = parser.parse_args()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    grid, summary = load_grid(args.input, args.normal_exaggeration)
    camera = camera_for(grid)
    sand_path = args.output_stem.parent / "sand_filled_raw.png"
    clay_path = args.output_stem.parent / "clay_smear_raw.png"
    render_panel(
        grid,
        subset(grid, clay=False),
        SAND,
        sand_path,
        camera,
        show_coordinate_system=True,
    )
    render_panel(
        grid,
        subset(grid, clay=True),
        CLAY,
        clay_path,
        camera,
        show_coordinate_system=False,
    )
    compose(sand_path, clay_path, args.output_stem, summary)

    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")
    print(summary)


if __name__ == "__main__":
    main()
