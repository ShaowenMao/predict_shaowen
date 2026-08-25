"""Render fine-scale PREDICT permeability and porosity in one fault core.

The permeability panel can use a selected local tensor component, expressed
as log10(k / mD). Both panels use the same realization, camera, geometry,
fault-normal exaggeration, and fault-local coordinate frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from PIL import Image
from scipy.io import loadmat


MD_IN_M2 = 9.869233e-16
OUTLINE = "#252525"
PHI_CLIM = (0.15, 0.27)
COMPONENTS = {
    "kxx": {"column": 0, "title": "Fault-normal permeability", "clim": (-5.2, 2.5)},
    "kyy": {"column": 3, "title": "Along-strike permeability", "clim": (-5.2, 2.7)},
    "kzz": {"column": 5, "title": "Dip-parallel permeability", "clim": (-5.0, 2.6)},
}


def load_grid(
    path: Path, normal_exaggeration: float, component: str
) -> tuple[pv.StructuredGrid, dict]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    coords = np.asarray(data["coords"], dtype=float)
    dims = np.asarray(data["cartDims"], dtype=int).ravel()
    perm = np.asarray(data["perm"], dtype=float)
    poro = np.asarray(data["poro"], dtype=float).ravel(order="F")

    n_cells = int(np.prod(dims))
    if coords.shape != (int(np.prod(dims + 1)), 3):
        raise ValueError(f"Unexpected replay-grid geometry: {coords.shape}")
    if perm.shape != (n_cells, 6) or poro.size != n_cells:
        raise ValueError(
            f"Unexpected property arrays: perm={perm.shape}, poro={poro.shape}, cells={n_cells}"
        )

    component_info = COMPONENTS[component]
    permeability_md = perm[:, component_info["column"]] / MD_IN_M2
    if np.any(~np.isfinite(permeability_md)) or np.any(permeability_md <= 0):
        raise ValueError(f"{component} permeability must be finite and positive.")
    if np.any(~np.isfinite(poro)):
        raise ValueError("Porosity must be finite.")

    points = coords.copy()
    points[:, 0] *= normal_exaggeration
    # PREDICT local z is positive down-dip; display positive z downward.
    points[:, 2] *= -1.0

    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = dims + 1
    grid.cell_data["log10_permeability_md"] = np.log10(permeability_md)
    grid.cell_data["porosity"] = poro

    summary = {
        "dims": dims.tolist(),
        "normal_exaggeration": normal_exaggeration,
        "permeability_component": component,
        "permeability_md_range": [
            float(permeability_md.min()),
            float(permeability_md.max()),
        ],
        "log10_permeability_md_range": [
            float(np.log10(permeability_md).min()),
            float(np.log10(permeability_md).max()),
        ],
        "porosity_range": [float(poro.min()), float(poro.max())],
        "physical_bounds_m": np.vstack((coords.min(axis=0), coords.max(axis=0))).tolist(),
    }
    return grid, summary


def camera_for(grid: pv.StructuredGrid):
    center = np.array(grid.center)
    length = grid.length
    position = center + np.array([0.72, -1.15, 0.52]) * length
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


def add_local_coordinate_system(plotter: pv.Plotter, grid: pv.StructuredGrid) -> None:
    """Add the PREDICT fault-local axes: x normal, y strike, z down dip."""
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    span_y = ymax - ymin
    span_z = zmax - zmin
    length = 0.085 * min(span_y, span_z)
    origin = np.array(
        [xmin - 0.08 * length, ymin - 0.18 * span_y, zmin + 0.16 * span_z],
        dtype=float,
    )
    directions = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, -1.0]),
    }
    tips, labels = [], []
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
    plotter.add_mesh(
        pv.Sphere(radius=0.030 * length, center=origin), color=OUTLINE, lighting=False
    )
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
    scalar: str,
    cmap: str,
    clim: tuple[float, float],
    output: Path,
    camera,
    show_coordinate_system: bool,
) -> None:
    plotter = pv.Plotter(off_screen=True, window_size=(1800, 1900))
    plotter.set_background("white")
    shell = grid.extract_surface(algorithm="dataset_surface")
    plotter.add_mesh(
        shell,
        scalars=scalar,
        preference="cell",
        cmap=cmap,
        clim=clim,
        opacity=1.0,
        show_edges=True,
        edge_color="#3A3735",
        line_width=0.40,
        lighting=False,
        show_scalar_bar=False,
    )
    outline = shell.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    plotter.add_mesh(outline, color=OUTLINE, line_width=2.35, lighting=False)
    if show_coordinate_system:
        add_local_coordinate_system(plotter, grid)
    plotter.camera_position = camera
    plotter.enable_parallel_projection()
    plotter.camera.zoom(0.76)
    plotter.show(screenshot=str(output), auto_close=True)


def crop_rgba(image: Image.Image, pad: int = 22) -> Image.Image:
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


def compose(
    k_path: Path,
    phi_path: Path,
    output_stem: Path,
    component: str,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.linewidth": 0.7,
        }
    )
    images = [crop_rgba(Image.open(k_path)), crop_rgba(Image.open(phi_path))]
    fig = plt.figure(figsize=(7.0, 4.55))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.055),
        left=0.025,
        right=0.985,
        bottom=0.105,
        top=0.955,
        wspace=0.125,
        hspace=0.035,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
    caxes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    component_info = COMPONENTS[component]
    titles = [component_info["title"], "Porosity"]
    letters = ["(a)", "(b)"]
    for ax, image, title, letter in zip(axes, images, titles, letters):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, pad=2.5, fontsize=11)
        ax.text(0.012, 0.988, letter, transform=ax.transAxes, ha="left", va="top")

    k_clim = component_info["clim"]
    k_norm = mpl.colors.Normalize(vmin=k_clim[0], vmax=k_clim[1])
    p_norm = mpl.colors.Normalize(vmin=PHI_CLIM[0], vmax=PHI_CLIM[1])
    cbk = fig.colorbar(
        mpl.cm.ScalarMappable(norm=k_norm, cmap="magma"),
        cax=caxes[0],
        orientation="horizontal",
        ticks=[-5, -3, -1, 1, k_clim[1]],
    )
    cbk.set_label(rf"$\log_{{10}}(k_{{{component[1:]}}}/\mathrm{{mD}})$", labelpad=1.5)
    cbp = fig.colorbar(
        mpl.cm.ScalarMappable(norm=p_norm, cmap="viridis"),
        cax=caxes[1],
        orientation="horizontal",
        ticks=[0.15, 0.18, 0.21, 0.24, 0.27],
    )
    cbp.set_label(r"Porosity, $\phi$", labelpad=1.5)
    for cax in caxes:
        cax.tick_params(direction="in", length=2.5, width=0.6, pad=1.5)

    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--normal-exaggeration", type=float, default=22.0)
    parser.add_argument("--component", choices=tuple(COMPONENTS), default="kxx")
    args = parser.parse_args()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    grid, summary = load_grid(args.input, args.normal_exaggeration, args.component)
    camera = camera_for(grid)
    k_path = args.output_stem.parent / f"{args.component}_permeability_raw.png"
    phi_path = args.output_stem.parent / "porosity_raw.png"
    render_panel(
        grid,
        "log10_permeability_md",
        "magma",
        COMPONENTS[args.component]["clim"],
        k_path,
        camera,
        True,
    )
    render_panel(grid, "porosity", "viridis", PHI_CLIM, phi_path, camera, False)
    compose(k_path, phi_path, args.output_stem, args.component)

    print(f"Wrote {args.output_stem.with_suffix('.png')}")
    print(f"Wrote {args.output_stem.with_suffix('.pdf')}")
    print(summary)


if __name__ == "__main__":
    main()
