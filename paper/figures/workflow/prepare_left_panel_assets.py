"""Prepare transparent, tightly cropped assets for the workflow figure.

The source scientific graphics are retained in ``assets``. This script removes
only their opaque white page backgrounds and writes reproducible derivatives to
``assets/derived``. Four source fault-core realizations are processed separately
so LaTeX can display them at one common visual height.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pikepdf
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RAW = ASSETS / "raw"
DERIVED = ASSETS / "derived"


def _remove_white_background(
    image: Image.Image,
    *,
    edge_connected_only: bool,
    white_threshold: int = 248,
    padding: int = 3,
) -> Image.Image:
    """Remove white background pixels and crop to the remaining content.

    ``edge_connected_only`` preserves enclosed white features, such as the
    stratigraphic separators in a fault-core realization, while removing the
    surrounding white page.
    """
    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[:, :, :3]
    white = np.all(rgb >= white_threshold, axis=2)

    if edge_connected_only:
        seed = np.zeros_like(white, dtype=bool)
        seed[0, :] = white[0, :]
        seed[-1, :] = white[-1, :]
        seed[:, 0] = white[:, 0]
        seed[:, -1] = white[:, -1]
        background = ndimage.binary_propagation(seed, mask=white)
    else:
        background = white

    rgba[background, 3] = 0
    result = Image.fromarray(rgba)
    bbox = result.getchannel("A").getbbox()
    if bbox is None:
        raise RuntimeError("Background removal erased the entire image.")

    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(result.width, right + padding)
    bottom = min(result.height, bottom + padding)
    return result.crop((left, top, right, bottom))


def _save(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, dpi=(600, 600), optimize=True)


def _remove_white_vector_background(
    source: Path,
    destination: Path,
    *,
    crop_box: tuple[float, float, float, float] | None = None,
) -> None:
    """Remove white PDF fills while preserving vector text, axes, and curves.

    ``crop_box`` uses PDF coordinates ``(left, bottom, right, top)`` and is
    useful for extracting a clean scientific panel from a larger source figure.
    """
    path_operators = {"m", "l", "c", "v", "y", "h", "re"}
    fill_operators = {"f", "F", "f*"}
    paint_operators = fill_operators | {"S", "s", "B", "B*", "b", "b*", "n"}

    pdf = pikepdf.Pdf.open(source)
    for page in pdf.pages:
        if crop_box is not None:
            box = pikepdf.Array(crop_box)
            page.MediaBox = box
            page.CropBox = box

        filtered = []
        pending_path = []
        white_fill = False

        for operands, operator in pikepdf.parse_content_stream(page):
            name = str(operator)
            if name == "rg":
                white_fill = len(operands) == 3 and all(
                    abs(float(value) - 1.0) < 1e-9 for value in operands
                )

            if name in path_operators:
                pending_path.append((operands, operator))
                continue

            if name in paint_operators:
                if white_fill and name in fill_operators:
                    pending_path.clear()
                    continue
                filtered.extend(pending_path)
                pending_path.clear()
                filtered.append((operands, operator))
                continue

            if pending_path:
                filtered.extend(pending_path)
                pending_path.clear()
            filtered.append((operands, operator))

        filtered.extend(pending_path)
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(filtered))

    destination.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(destination)


def _is_current(destination: Path, source: Path) -> bool:
    """Return true when a derivative is newer than its source and this script."""
    if not destination.exists():
        return False
    newest_input = max(source.stat().st_mtime, Path(__file__).stat().st_mtime)
    return destination.stat().st_mtime >= newest_input


def prepare() -> None:
    """Regenerate every transparent derivative used by the workflow figure."""
    for index in range(1, 5):
        source = RAW / f"fault_architecture_{index:02d}.png"
        destination = DERIVED / f"fault_architecture_{index:02d}.png"
        if _is_current(destination, source):
            continue
        with Image.open(source) as image:
            transparent = _remove_white_background(
                image,
                edge_connected_only=True,
                white_threshold=245,
            )
        _save(transparent, destination)

    for source_name, output_name in (
        ("permeability_distributions.pdf", "permeability_distributions.pdf"),
        ("pc_kr_curves.pdf", "pc_kr_curves.pdf"),
    ):
        source = ASSETS / source_name
        destination = DERIVED / output_name
        if _is_current(destination, source):
            continue
        _remove_white_vector_background(source, destination)

    # Extract one representative six-window field from each larger source
    # figure. Keeping only the heatmap makes these summaries legible at the
    # final three-column manuscript size.
    for source_name, output_name, crop_box in (
        (
            "full_fault_kzz.pdf",
            "full_fault_kzz_map.pdf",
            (510.0, 224.0, 1275.0, 280.0),
        ),
        (
            "full_fault_phi.pdf",
            "full_fault_phi_map.pdf",
            (70.0, 115.0, 1410.0, 210.0),
        ),
    ):
        source = ASSETS / source_name
        destination = DERIVED / output_name
        if _is_current(destination, source):
            continue
        _remove_white_vector_background(
            source,
            destination,
            crop_box=crop_box,
        )

    # These raster derivatives were used by an earlier draft. Removing them
    # avoids accidental reuse of blurred text in the final manuscript figure.
    for stale_name in (
        "permeability_distributions.png",
        "pc_kr_curves.png",
        "full_fault_kzz.pdf",
        "full_fault_phi.pdf",
    ):
        (DERIVED / stale_name).unlink(missing_ok=True)


if __name__ == "__main__":
    prepare()
