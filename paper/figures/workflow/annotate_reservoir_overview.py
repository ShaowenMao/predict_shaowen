"""Add publication-style geological annotations to the full reservoir render."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets" / "derived" / "reservoir_full_unannotated.png"
OUTPUT = ROOT / "assets" / "derived" / "reservoir_full_annotated.png"
FONT_PATH = Path(r"C:\Windows\Fonts\times.ttf")


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: str = "#111111",
    width: int = 5,
    head_length: int = 20,
    head_width: int = 16,
) -> None:
    """Draw a clean leader line with a filled arrowhead."""
    draw.line((start, end), fill=fill, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    base_x = end[0] - head_length * math.cos(angle)
    base_y = end[1] - head_length * math.sin(angle)
    normal_x = head_width * math.sin(angle) / 2
    normal_y = -head_width * math.cos(angle) / 2
    draw.polygon(
        (
            end,
            (base_x + normal_x, base_y + normal_y),
            (base_x - normal_x, base_y - normal_y),
        ),
        fill=fill,
    )


def add_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    text: str,
    text_xy: tuple[int, int],
    arrow_start: tuple[int, int],
    target: tuple[int, int],
    *,
    anchor: str = "la",
    align: str = "left",
) -> None:
    """Draw one label and its leader arrow."""
    draw.multiline_text(
        text_xy,
        text,
        font=font,
        fill="#111111",
        anchor=anchor,
        align=align,
        spacing=2,
    )
    draw_arrow(draw, arrow_start, target)


def main() -> None:
    """Annotate the fixed full-reservoir view without altering the base render."""
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if not FONT_PATH.is_file():
        raise FileNotFoundError(FONT_PATH)

    with Image.open(SOURCE) as source:
        image = source.convert("RGBA")
    width, height = image.size
    draw = ImageDraw.Draw(image)
    # At the 45 mm panel width, 96 source pixels render close to the workflow's
    # normal text size while remaining legible after PDF downsampling.
    font = ImageFont.truetype(str(FONT_PATH), size=96)

    # Keep leader arrows short and orthogonal so they do not cross one another.
    add_label(
        draw,
        font,
        "Main fault",
        (round(0.47 * width), round(0.105 * height)),
        (round(0.47 * width), round(0.22 * height)),
        (round(0.47 * width), round(0.34 * height)),
        anchor="ma",
        align="center",
    )
    add_label(
        draw,
        font,
        "Top seal",
        (round(0.84 * width), round(0.22 * height)),
        (round(0.84 * width), round(0.34 * height)),
        (round(0.84 * width), round(0.41 * height)),
        anchor="ma",
        align="center",
    )
    add_label(
        draw,
        font,
        "Storage\nreservoir",
        (round(0.84 * width), round(0.69 * height)),
        (round(0.84 * width), round(0.66 * height)),
        (round(0.84 * width), round(0.52 * height)),
        anchor="ma",
        align="center",
    )
    add_label(
        draw,
        font,
        "Top-seal\nfault region",
        (round(0.17 * width), round(0.43 * height)),
        (round(0.37 * width), round(0.50 * height)),
        (round(0.49 * width), round(0.50 * height)),
        anchor="mm",
        align="center",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, dpi=(600, 600), optimize=True)
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
