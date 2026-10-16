# Synchronized MIP saturation, Pc and Kr movies

This standalone workflow reproduces the approved Core 172 animation and applies
the same presentation to seven other saved representative cores. It does not
import manuscript scripts, production managers or local study-directory code.
It never changes production results or submits follow-on jobs.

## Scientific meaning

Each frame is one of the original 100 applied gas-brine pressure differences,
not a physical time step. Fine-cell gas saturation is reconstructed by a
deterministic MIP replay from the saved input. These cell fields are reconstructed,
not claimed to be archived original fields. Before rendering, all pressure,
bulk pore-volume saturation, trapped/immobile/invaded pore-volume fractions,
breakthrough flags and event counts must match the saved run. Gas and brine Kr
are the original calculated values; there are **no new pressure-flow solves**.
Native reservoir-input Kr tables are independently checked against those values.

The pore-volume mean is `sum(Sg * porosity * cell_volume) / sum(porosity * cell_volume)`.
Sand and clay labels use the same formula restricted to that material. Both views
retain every original cell, but interior cells can be occluded by exposed surfaces.
The core is stretched 22 times in thickness for visibility; local positive z is
updip, from the bottom gas inlet toward the top outlet. The x-y-z marker is
projected from the transformed grid using the actual orthographic camera.

## Approved presentation

- 3840 x 2160 H.264 movie and 1440 x 810 infinite-loop GIF.
- 100 exact states at 0.1 s/state, plus 1 s initial and 2 s final holds: 13 s total.
- Real LaTeX Computer Modern, main text 20 pt; gray annotations 12.5 and 10.3 pt.
- Fixed-position applied pressure and bulk saturation fields; no moving vertical guide.
- Pc above Kr, equal-sized panels, inward ticks, no grid, two-decimal line-plot ticks.
- Black Pc, orange gas Kr and solid blue brine Kr; original shared saturation scale.
- Half-length colorbar with only 0.0 and 1.0 ticks; material titles black.
- Header/material row spacing is 1.5 times the previous 0.042 figure-height gap.
- Small padding around the Kr axes; Pc upper bound expands only when necessary
  to avoid clipping a different core's pressure range. No curve rescaling/smoothing.
- Camera zoom fits both projected dimensions to each material's actual viewport;
  wider cores are not cropped. Core 172 retains its approved scale and angle.

The GIF endpoint pauses are inserted by changing delay bytes only, without a
second palette quantization. Do not re-save it with Pillow just to change timing.

## Selected cores

| Risk | Case | Cores |
|---|---|---|
| Low | s01_c018_case07 | 143, 462 |
| Intermediate | s03_c022_case12 | 98, 100, 167 |
| High | s03_c003_case08 | 144, 165, 172 |

Core 172 is the reference/pilot. `render_keys.txt` contains the other seven.
These examples illustrate contrasting responses, not frequencies or controlled
experiments. Large binary inputs, frames and movies remain outside Git.

## Input preparation (once, local or on compute)

Use Python 3.12 or newer and `requirements.txt`. The archived audit contains
`raw.json`, `analysis.json`, and `fine_inputs/<risk>_<slot>[_geometry].bin`.
The native directory contains the three `<risk>_mip_wrr60_fault_nohys.mat` files.

```sh
python -B prepare_bundle.py --audit /path/to/audit --native /path/to/native --output /new/bundle
python -B -m unittest -v test_workflow
bash test_launcher.sh
```

The bundle contains per-core original raw input, TPFA geometry, saved accepted
record and extracted native Kr table, plus a SHA-256 manifest. Every topology
edge, inlet/outlet set, geometry parent hash, cell spacing and material-cell ID
is checked. A failed scientific check stops only that independent render.

For the optional regression against the approved Core 172 movie data, set
`MIPVIZ_BUNDLE` and `MIPVIZ_BASELINE` before running the tests. The original baseline
is `mip_kr_core172_animation_20261015_v1` under the external studies directory.

## Engaging execution

Connect with plain interactive `ssh engaging`. Stage this directory as
`RUN/workflow`, the frozen bundle as `RUN/bundle`, and create `RUN/logs`.
Use a dedicated run directory under `qualification_visuals`, never production.
The supplied environment/setup and render scripts explicitly request
`mit_normal`, account **and** QoS `mit_amf_advanced_cpu`. No general fallback.

```sh
sbatch --hold --output="$RUN/logs/setup-%j.out" --error="$RUN/logs/setup-%j.err" \
  "$RUN/workflow/setup_env.sh" "$RUN"
# Inspect scontrol show job JOB: account, QoS, partition, CPUs, memory, launcher.
# Only then: scontrol release JOB. Wait for successful setup.
sbatch --hold --output="$RUN/logs/pilot-%j.out" --error="$RUN/logs/pilot-%j.err" \
  "$RUN/workflow/render_job.sh" "$RUN" high_172
# Verify allocation/launcher, release, and wait for COMPLETE.json and media_qa PASS.
# Inspect peak memory and elapsed time before choosing the seven-task allocation.
sbatch --hold --array=0-6 --output="$RUN/logs/render-%A_%a.out" \
  --error="$RUN/logs/render-%A_%a.err" "$RUN/workflow/render_job.sh" "$RUN"
# Verify every array task's request; no array concurrency throttle is specified.
# Release only this new rendering array. Leave all other jobs alone.
```

Slurm schedules seven independent tasks. There is no resident manager, refill
logic, ownership database, or dependence on unrelated job names. The pilot
requests 4 CPUs and 8192 MiB conservatively for CPU-only software rendering;
production rendering allocation is chosen from its measured peak plus headroom.
The setup job requests 2 CPUs/4096 MiB. Rendering uses Mesa/EGL, TeX Live and Julia
modules; no GPU is needed. Private LaTeX and Julia caches avoid concurrent writes.
Requirements are installed once in a dedicated environment on a compute node.

The array launcher accepts LF and CRLF task lists and rejects missing/invalid
keys before loading scientific code. `MIPVIZ_VALIDATE_ONLY=1` exercises that
launcher without starting a render. The launcher uses its own versioned source
directory from the third argument (default `RUN/workflow`), not from the Slurm
spool-script path. For a separate source version use
`render_job.sh RUN array /absolute/source/version`. A repair need not overwrite
the original deployed code. The launcher test includes a Slurm-like copied script.

`run_core.py` refuses an existing output directory. On failure retain the old
output and log; investigate and use an explicit new output/run directory. Do not
blindly resubmit an uncertain Slurm submission. Rendering never recalculates
accepted Pc/Kr flow results, even when retrying visualization-only work.

## Outputs and validation

`outputs/<risk>_<slot>/` contains the movie, GIF, 100 4K frames,
`saturation_states.bin`, `frame_values.csv`, `reconstruction.toml`,
`validation.json`, `style_checks.json`, `media_qa.json`, `timings.json` and
`COMPLETE.json`. Completion means scientific, layout, timing, full movie decode
and GIF pixel/palette checks passed, not just a successful Slurm exit.

The replay library in `reference/` is the unchanged algorithm snapshot used for
the Core 172 validation and earlier Core 144 trapping analysis. Only text line
endings are normalized. `reference/SOURCE.json` records provenance and hashes.
The renderer is derived from approved animation revision
`mip_kr_core172_animation_20261016_v6`, generalized only for input/output paths,
core labels and per-core pressure bounds. Scientific inputs and sampling are unchanged.
