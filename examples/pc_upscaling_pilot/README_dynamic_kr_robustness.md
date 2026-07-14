# Production Pc and Dynamic Kr Upscaling Notes

This folder contains the production-oriented full-87 upscaling workflow for
the representative median-sand-ratio Level-3 examples. The intended path is:

1. replay the exact selected PREDICT realizations;
2. upscale Pc with the invasion-percolation workflow;
3. upscale Kr with the dynamic workflow.

Fast proxy Pc workflows and capillary-limit Kr screening workflows were
removed from this folder so the codebase points to the rigorous path.

## Replay Preparation

Run:

```matlab
prepare_full87_replay_median_examples
```

This writes the replay table consumed by both production upscaling scripts:

```text
D:\codex_gom\UQ_workflow\full87_replay_median_examples\tables\
replay_summary_with_full87_context_s05_c012_cases_01_03_04_07.csv
```

Useful environment overrides:

- `FULL87_REPLAY_OUTPUT_ROOT`
- `FULL87_REPLAY_FIELD_SAMPLING_CSV`
- `FULL87_REPLAY_DATA_ROOT`
- `PREDICT_REPLAY_CODE_ROOT`
- `FULL87_REPLAY_MAX_ROWS`

## Pc Upscaling

Run:

```matlab
run_pc_upscaling_ip_median_examples_full87
```

This is the connectivity-based invasion-percolation Pc workflow. For a smoke
test:

```matlab
setenv('PC_IP_MAX_ROWS', '1')
run_pc_upscaling_ip_median_examples_full87
```

Clear `PC_IP_MAX_ROWS` for the full run.

Every native Pc curve is a production reservoir input. Full-curve medoids are
diagnostics only and are disabled by default. Enable them with:

```matlab
setenv('PC_IP_ENABLE_MEDOID_DIAGNOSTICS', '1')
```

## Dynamic Kr Upscaling

Run:

```matlab
run_kr_upscaling_dyn_median_examples_full87
```

The default behavior preserves the original 1D dynamic matching result:

- `KR_DYN_1D_METHOD=ad`
- `KR_DYN_1D_AD_SOLVER=legacy`

That means clean rows use the same AD-style 1D Corey matching as the
original workflow, with added progress logging and checkpoints.

For production reduction after all Pc curves are available, use:

```matlab
setenv('KR_DYN_SELECTION_MODE', 'swi_medoid')
setenv('KR_DYN_EXPORT_RESERVOIR_READY', '1')
```

This selects one actual scalar-Swi medoid realization per window, runs six
dynamic Kr calculations per Level-3 case, normalizes those shapes, restores
all 87 slice-specific Pc endpoints, and writes checked reservoir-ready MAT
files. Use `KR_DYN_SELECTION_MODE=all` only for full-87 validation.

## Robust Fallback Options

For rows that stall, use:

```powershell
$env:KR_DYN_FALLBACK_1D_AD_SOLVER = "robust"
.\run_kr_upscaling_dyn_supervised_full87.ps1
```

The robust AD fallback keeps the same 1D AD model family but uses an explicit
nonlinear solver with line search, iteration timestep strategy, and controlled
timestep cuts. This fixed the previously observed full-grid W6 candidate-15
stall.

An experimental transport matcher is also available:

```powershell
$env:KR_DYN_1D_METHOD = "transport"
```

Use it only for diagnostics or emergency fallback. It is robust, but it is not
numerically equivalent to the AD matcher in the smoke comparison.

## Supervised Process Runner

Run one MATLAB process per replay row:

```powershell
$env:KR_DYN_CURVE_TIMEOUT_SECONDS = "5400"
$env:KR_DYN_SUPERVISED_ROWS = "1-20"
.\run_kr_upscaling_dyn_supervised_full87.ps1
```

If a row exceeds the timeout, the supervisor kills that MATLAB process and can
retry the same row with fallback settings. Completed rows are saved as
per-curve checkpoints, so the workflow can resume without recomputing successful
curves.

## HPC Production Notes

The replay/Pc/Kr workflow is naturally suited to a scheduler job array. Use one
MATLAB process per replay row, or small row batches, and avoid nested
parallelism unless benchmarking shows a clear benefit.

Recommended starting point on Engaging:

- Use job arrays across rows.
- Set `KR_DYN_USE_PARALLEL=0` inside each row job.
- Set `KR_DYN_3D_NUM_THREADS=2` initially.
- Use `KR_DYN_CURVE_TIMEOUT_SECONDS=5400` for full-grid rows.
- Aggregate completed per-row checkpoints after the array finishes.

This keeps the workflow robust to runtime variability: slow Pc rows can be
resubmitted or given longer wall time without recomputing completed rows.

Useful options:

- `KR_DYN_SUPERVISED_ROWS`: row list or ranges, e.g. `1,6,10-20`.
- `KR_DYN_CURVE_TIMEOUT_SECONDS`: primary per-row timeout. Use at least
  `3600` seconds for full-grid stress rows; `5400` seconds is safer because
  Pc invasion-percolation can dominate runtime for some maps.
- `KR_DYN_FALLBACK_TIMEOUT_SECONDS`: fallback per-row timeout.
- `KR_DYN_FALLBACK_1D_AD_SOLVER`: usually `robust`.
- `KR_DYN_FALLBACK_TIMESTEP_MODE`: optional fallback timestep mode, e.g.
  `paper_fine`.

## 3D Dynamic Controls

The 3D dynamic solve now exposes:

- `KR_DYN_3D_MAX_ITER`
- `KR_DYN_3D_MAX_CUTS`
- `KR_DYN_3D_USE_LINESEARCH`
- `KR_DYN_3D_USE_RELAXATION`
- `KR_DYN_3D_NUM_THREADS`
- `KR_DYN_TIMESTEP_MODE=paper`, `smoke`, or `paper_fine`

`paper_fine` uses smaller report/control steps through the paper-length
simulation. It is intended as a robustness fallback for 3D timestep stalls.

## Validation Completed

- Cropped smoke run with legacy AD default reproduced the previous legacy
  fitted result exactly.
- Cropped smoke run with forced timeout verified process kill and fallback.
- Full-grid W6 row 6 with robust AD completed all 256 Corey candidates. The
  previously stalled candidate 15 completed in less than one second.
- Additional full-grid stress rows completed with robust AD: rows 252, 920,
  1566, and 1621. Row 1566 needed a longer timeout because Pc upscaling took
  about 27.6 minutes, but the dynamic 3D and 1D matching stages completed
  normally.
- Cropped `paper_fine` run completed, validating the finer 3D timestep option.
