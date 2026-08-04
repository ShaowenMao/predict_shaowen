"""Render the field-scale fault geometry from an MRST-exported VTU file.

The VTU ``fault_region_flag`` values are interpreted as documented by the
export workflow:

``0`` outside the fault, ``1`` PREDICT fault, and ``2`` non-PREDICT fault.

The resulting transparent PNG is intended for the middle panel of the
end-to-end workflow figure. It shows the true three-dimensional fault geometry
while highlighting the top-seal interval where PREDICT properties are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    r"D:\codex_gom\step62_effective_pc_global_plateau"
    r"\case5_s03_c001_case04_geology_v2"
    r"\gom_step62_effective_pc_global_plateau_"
    r"s03_c001_case04_geology_v2_0210.vtu"
)
DEFAULT_OUTPUT = ROOT / "assets" / "fault_geometry_3d.png"

FAULT_FLAG = "fault_region_flag"
PREDICT_FLAG = 1
NONPREDICT_FLAG = 2

PREDICT_COLOR = "#1769AA"
NONPREDICT_COLOR = "#A7ADB4"


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and rendering controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1500)
    parser.add_argument("--vertical-exaggeration", type=float, default=3.0)
    parser.add_argument(
        "--camera-along-strike",
        type=float,
        default=1.50,
        help="Along-strike camera offset relative to the fault-normal offset.",
    )
    parser.add_argument(
        "--camera-elevation",
        type=float,
        default=0.0,
        help="Vertical camera offset relative to the fault-normal offset.",
    )
    parser.add_argument(
        "--view-angle",
        type=float,
        default=24.0,
        help="Perspective camera view angle in degrees.",
    )
    parser.add_argument(
        "--projection",
        choices=("parallel", "perspective"),
        default="parallel",
        help="Camera projection; parallel is the publication-figure default.",
    )
    return parser.parse_args()


def extract_fault_surface(
    grid: pv.UnstructuredGrid,
    flag_value: int,
) -> pv.PolyData:
    """Extract the exterior surface of one fault-region class."""
    selected = grid.threshold(
        value=(flag_value - 0.25, flag_value + 0.25),
        scalars=FAULT_FLAG,
        preference="cell",
    )
    if selected.n_cells == 0:
        raise ValueError(f"No cells found for {FAULT_FLAG}={flag_value}")
    return selected.extract_surface(algorithm="dataset_surface").clean()


def transform_depth(
    surface: pv.PolyData,
    vertical_exaggeration: float,
) -> pv.PolyData:
    """Display positive input depth downward with controlled exaggeration."""
    transformed = surface.copy(deep=True)
    points = np.asarray(transformed.points).copy()
    points[:, 2] *= -vertical_exaggeration
    transformed.points = points
    return transformed


def crop_transparent_image(path: Path, padding: int = 36) -> None:
    """Remove the white render background and crop transparent margins."""
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA")).copy()
        rgb = rgba[:, :, :3].astype(np.float64)

        # Some Windows VTK backends return an opaque white background even
        # when transparent capture is requested. Convert distance from white
        # into alpha, retaining antialiased object edges without a white halo.
        distance_from_white = 255.0 - np.min(rgb, axis=2)
        alpha = np.clip(5.0 * distance_from_white, 0.0, 255.0)
        rgba[:, :, 3] = alpha.astype(np.uint8)
        image = Image.fromarray(rgba)
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            raise RuntimeError("VTK rendered a fully transparent image")
        left, top, right, bottom = bbox
        crop_box = (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
        cropped = image.crop(crop_box)
        cropped.save(path, dpi=(600, 600), optimize=True)


def render(args: argparse.Namespace) -> None:
    """Render the PREDICT and non-PREDICT fault regions."""
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.vertical_exaggeration <= 0:
        raise ValueError("vertical exaggeration must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid = pv.read(args.input)
    if FAULT_FLAG not in grid.cell_data:
        raise KeyError(f"{FAULT_FLAG!r} is missing from {args.input}")

    flags = np.asarray(grid.cell_data[FAULT_FLAG])
    observed = set(np.unique(flags).tolist())
    required = {0, PREDICT_FLAG, NONPREDICT_FLAG}
    if not required.issubset(observed):
        raise ValueError(
            f"Expected {FAULT_FLAG} values {sorted(required)}, "
            f"found {sorted(observed)}"
        )

    predict = transform_depth(
        extract_fault_surface(grid, PREDICT_FLAG),
        args.vertical_exaggeration,
    )
    nonpredict = transform_depth(
        extract_fault_surface(grid, NONPREDICT_FLAG),
        args.vertical_exaggeration,
    )

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
        smooth_shading=True,
        ambient=0.34,
        diffuse=0.78,
        specular=0.12,
        specular_power=18.0,
    )
    plotter.add_mesh(
        predict,
        color=PREDICT_COLOR,
        smooth_shading=True,
        ambient=0.30,
        diffuse=0.82,
        specular=0.18,
        specular_power=22.0,
    )

    combined_bounds = np.asarray(
        [
            min(predict.bounds.x_min, nonpredict.bounds.x_min),
            max(predict.bounds.x_max, nonpredict.bounds.x_max),
            min(predict.bounds.y_min, nonpredict.bounds.y_min),
            max(predict.bounds.y_max, nonpredict.bounds.y_max),
            min(predict.bounds.z_min, nonpredict.bounds.z_min),
            max(predict.bounds.z_max, nonpredict.bounds.z_max),
        ]
    )
    center = np.asarray(
        [
            np.mean(combined_bounds[0:2]),
            np.mean(combined_bounds[2:4]),
            np.mean(combined_bounds[4:6]),
        ]
    )
    extent = np.asarray(
        [
            combined_bounds[1] - combined_bounds[0],
            combined_bounds[3] - combined_bounds[2],
            combined_bounds[5] - combined_bounds[4],
        ]
    )

    # View from the hanging-wall side. The along-strike offset exposes the
    # listric down-dip profile at the near end, while the small elevation and
    # fixed vertical view-up keep strike approximately horizontal.
    camera_distance = float(np.linalg.norm(extent))
    plotter.camera_position = [
        center
        + camera_distance
        * np.asarray(
            [args.camera_along_strike, -1.0, args.camera_elevation]
        ),
        center,
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.parallel_projection = args.projection == "parallel"
    plotter.camera.view_angle = args.view_angle
    plotter.reset_camera()
    plotter.camera.zoom(1.12)

    plotter.show(auto_close=False)
    try:
        plotter.screenshot(
            str(args.output),
            transparent_background=True,
            return_img=False,
        )
    finally:
        plotter.close()
    crop_transparent_image(args.output)

    counts = {
        int(value): int(np.count_nonzero(flags == value))
        for value in sorted(required)
    }
    print(f"Input: {args.input}")
    print(f"Fault cell counts: {counts}")
    print(f"Vertical exaggeration: {args.vertical_exaggeration:g}x")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    render(parse_args())
