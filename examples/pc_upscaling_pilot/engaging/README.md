# Engaging Case01 Workflow

This folder contains the first Engaging workflow for the rigorous production
path:

1. replay selected PREDICT realizations;
2. run invasion-percolation Pc upscaling;
3. run dynamic Kr upscaling.

The initial target is the previously reviewed median-sand, nonuniform case01
example:

- geology id: `s05_c012`
- Level-3 case id: `1`
- windows: `famp1` to `famp6`
- slices: 87 along-strike slices in full mode

## Prerequisites

MRST 2025a is installed on Engaging at:

```text
/home/shaowen/orcd/pool/predict_shaowen/software/mrst-current
```

The workflow also expects:

```text
/home/shaowen/orcd/pool/predict_shaowen/repo
/home/shaowen/orcd/pool/predict_shaowen/software/upscaling.zip
/home/shaowen/orcd/pool/predict_shaowen/inputs/texas_offshore_field_sampling/texas_field_slice_window_values.csv
```

AMGCL should be compiled before dynamic-Kr production runs. This is required
on Engaging because the full 3D dynamic solves are otherwise dominated by slow
linear solves:

```bash
cd /home/shaowen/orcd/pool/predict_shaowen/repo/examples/pc_upscaling_pilot/engaging
bash setup_mrst_amgcl.sh
```

The script installs MRST's pinned AMGCL revision
`4f260881c7158bc5aede881f5f0ed272df2ab580`, rebuilds
`amgcl_matlab.mexa64` and `amgcl_matlab_block.mexa64`, and runs a small
Poisson-system smoke test. The full Case01 environment sets
`KR_DYN_LINEAR_SOLVER=amgcl_require`, so the Kr stage fails early if AMGCL is
not available instead of silently using a slow fallback.

High-I/O outputs go to:

```text
/home/shaowen/orcd/scratch/predict_shaowen/runs/<run_id>
```

## Smoke Test

Submit a small chained test:

```bash
cd /home/shaowen/orcd/pool/predict_shaowen/repo/examples/pc_upscaling_pilot/engaging
bash submit_case01_smoke.sh
```

Smoke mode replays only a few rows and runs tiny Pc/Kr calculations. It is
intended to validate paths, MATLAB/MRST loading, replay, and Slurm dependency
logic before full production.

## Full Case01

After the smoke test passes, use the same scripts with:

```bash
export RUN_MODE=full
export RUN_ID=case01_full_001
bash submit_case01_smoke.sh
```

Default full-mode resources are:

```text
replay: 3 hours, 1 CPU, 16 GB
Pc:     3 hours, 1 CPU, 24 GB
Kr:     12 hours, 12 CPUs, 96 GB
```

The dynamic-Kr full run always uses the full 3D replay grids, 12 MATLAB
process workers, one numerical thread per worker, and `KR_DYN_PC_PRESTEP_MODE`
set to `precomputed`. This parallelizes over independent replay rows while
avoiding nested CPU oversubscription, and it reuses the completed optimized Pc
curves plus their true endpoint summary instead of rerunning the slow internal
Pc pre-step inside the Kr stage.

To force the original internal Pc pre-step for debugging, use:

```bash
export KR_DYN_PC_PRESTEP_MODE=original
```

To require a completed Pc table and fail if it is missing, use:

```bash
export KR_DYN_PC_PRESTEP_MODE=precomputed
```

Strike-collapsed grids are not supported in this production workflow. The 87
along-strike slices are treated as real field heterogeneity, so each dynamic Kr
curve is computed from its replayed full 3D map.

Summarize a run with:

```bash
bash summarize_case01_run.sh case01_full_001
```

The summary reports Slurm status, stage-level wall time/memory diagnostics,
output row counts, and dynamic-Kr timing quantiles.

Full production for all 162 geologies should use the same design, but with a
manifest-driven job array rather than a single chained case01 job.
