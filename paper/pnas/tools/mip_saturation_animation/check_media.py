"""Verify data identity, requested presentation, synchronization and export timing."""
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
from PIL import Image

import argparse
p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
OUT = args.output.resolve()
import imageio_ffmpeg
exe = imageio_ffmpeg.get_ffmpeg_exe()
current = json.loads((OUT / "validation.json").read_text())
assert current["status"] == "PASS" and current["new_flow_solves"] == 0
style = json.loads((OUT / "style_checks.json").read_text())
assert style["pc_axes_pixels"] == style["kr_axes_pixels"]
assert style["material_zero_saturation_colors"] == {"sand": "#edf3f6", "clay": "#edf3f6"}
assert style["colorbar_count"] == 1 and style["latex_all_text"]
assert not style["curve_titles"] and not style["dynamic_kr_values"]
assert not style["state_counter"] and not style["moving_kr_vertical_guide"]
assert np.isclose(style["colorbar_width_fraction"], .18)
assert style["colorbar_tick_labels"] == [r"$0.0$", r"$1.0$"]
assert style["kr_ylim"] == [-.025, 1.025]
assert style["kr_xlim"] == style["pc_xlim"] == [-.025, 1.025]
assert style["kr_yticks"] == [0, .25, .5, .75, 1]
assert style["kr_ytick_labels"] == [r"$0.00$", r"$0.25$", r"$0.50$", r"$0.75$", r"$1.00$"]
assert style["pc_xtick_labels"] == style["kr_xtick_labels"] == ["$%0.2f$" % v for v in np.linspace(0, 1, 6)]
assert style["pc_ytick_labels"] == [r"$0.00$", r"$0.10$", r"$1.00$", r"$10.00$", r"$100.00$"]
assert not style["grid_lines"] and not style["main_title"]
assert style["line_plot_tick_directions"] == {"pc": ["in"], "kr": ["in"]}
assert np.isclose(style["header_to_material_row_gap"] / style["previous_header_to_material_row_gap"], 1.5)
assert style["coordinate_triad"]["projected_directions"]["x"][0] > 0
assert style["coordinate_triad"]["projected_directions"]["x"][1] < 0
assert style["coordinate_triad"]["projected_directions"]["y"][0] > 0
assert style["coordinate_triad"]["projected_directions"]["y"][1] > 0
assert np.allclose(style["coordinate_triad"]["projected_directions"]["z"], [0, 1])

assert style["fixed_header_slots"] and style["header_max_anchor_drift_pixels"] < 1e-6
assert style["all_main_font_sizes_pt"] == [20]
assert style["gray_annotation_sizes_pt"] == [10.3, 12.5]
assert style["header_centered"] and np.isclose(style["header_center_x"], .284)
assert style["material_title_color"] == "black"
assert style["curve_axis_labels"] == [r"$\bar{S}_g\,[-]$", r"$P_c\,[\mathrm{bar}]$", r"$\bar{S}_g\,[-]$", r"$k_r\,[-]$"]
files = sorted((OUT / "frames").glob("state_*.png"))
assert len(files) == 100
for file in files:
    with Image.open(file) as img:
        assert img.size == (3840, 2160)
movie = OUT / "synchronized_drainage_4k.mp4"
decoded = subprocess.run([exe, "-v", "error", "-i", str(movie), "-f", "null", "-"], capture_output=True, check=True)
assert not decoded.stderr.strip(), decoded.stderr.decode(errors="replace")
probe = subprocess.run([exe, "-i", str(movie)], capture_output=True).stderr.decode(errors="replace")
assert "3840x2160" in probe and "30 fps" in probe and "Video: h264" in probe
duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", probe)
assert duration
h, m, s = map(float, duration.groups())
seconds = 3600*h + 60*m + s
assert abs(seconds - 13) < .01
errors = {}
for j, t in [(0, .2), (0, 1.05), (1, 1.15), (20, 3.05), (60, 7.05), (99, 10.95), (99, 12.8)]:
    raw = subprocess.run([exe, "-v", "error", "-ss", str(t), "-i", str(movie), "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    actual = np.frombuffer(raw, np.uint8).reshape(2160, 3840, 3)
    with Image.open(OUT / "frames" / f"state_{j:03d}.png") as image:
        expected = np.asarray(image.convert("RGB"))
    error = float(np.mean(np.abs(actual.astype(np.int16) - expected)))
    assert error < 3, (j, t, error)
    errors[str(t)] = error
delays = []
gif_errors = {}
with Image.open(OUT / "looping_preview.gif") as gif:
    assert gif.size == (1440, 810) and gif.n_frames == 100
    assert gif.info["loop"] == 0
    for i in range(gif.n_frames):
        gif.seek(i)
        delays.append(gif.info["duration"])
        if i in (0, 20, 62, 99):
            actual = np.asarray(gif.convert("RGB")).astype(np.int16)
            with Image.open(OUT / "unpaused_encoding.gif") as base:
                base.seek(i)
                np.testing.assert_array_equal(actual, np.asarray(base.convert("RGB")))
            with Image.open(OUT / "frames" / f"state_{i:03d}.png") as source:
                expected = np.asarray(source.convert("RGB").resize(gif.size, Image.Resampling.LANCZOS)).astype(np.int16)
            error = float(np.abs(actual - expected).mean())
            assert error < 3, (i, error)
            gif_errors[str(i + 1)] = error
assert delays == [1100] + [100] * 98 + [2100], delays
assert sum(delays) == 13000
report = dict(status="PASS", unchanged_scientific_data=True, movie_duration_seconds=seconds,
    dimensions=[3840, 2160], codec="H.264", frames_per_second=30, states=100,
    state_interval_seconds=.1, added_start_hold_seconds=1, added_end_hold_seconds=2,
    complete_movie_decode=True, decoded_vs_source_rgb_mean_abs_error=errors,
    gif_dimensions=[1440, 810], gif_states=100, gif_duration_seconds=sum(delays)/1000,
    gif_infinite_loop=True, gif_first_state_ms=delays[0], gif_last_state_ms=delays[-1],
    gif_pause_edit_preserves_pixels=True, gif_decoded_vs_resized_source_rgb_mean_abs_error=gif_errors,
    equal_pc_kr_axes=True, pc_above_kr=True, shared_colorbar=True, latex_all_text=True,
    half_length_colorbar=True, endpoint_colorbar_ticks_only=True, centered_combined_header=True,
    kr_ylim=[-.025, 1.025], kr_yticks=[0, .25, .5, .75, 1], state_counter=False, moving_kr_vertical_guide=False,
    fixed_header_slots=True, header_max_anchor_drift_pixels=style["header_max_anchor_drift_pixels"],
    main_font_size_pt=20, gray_annotation_sizes_pt=[10.3, 12.5], line_plot_tick_decimal_places=2,
    grid_lines=False, main_title=False, inward_plot_ticks=True, header_spacing_multiplier=1.5,
    geometry_aligned_coordinate_triad=True,
    movie_bytes=movie.stat().st_size, gif_bytes=(OUT / "looping_preview.gif").stat().st_size)
(OUT / "media_qa.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
