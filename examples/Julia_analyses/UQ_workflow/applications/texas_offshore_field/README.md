# Texas Offshore Field Sampling Application

This folder applies the general three-level UQ workflow to the Texas offshore
GOM field model with 87 along-strike vertical fault slices.

The core Level 2 and Level 3 workflow remains unchanged. This application
reloads the existing Level 2 state objects and the deterministic Level 3
multiple-window case assignments, then repeats those assignments over field
slices.

For each geology:

- the same 10 multiple-window case designs are used on every slice;
- each slice draw group independently samples actual joint PREDICT
  realizations from the assigned Level 2 pools;
- all 87 field slices use independent draw groups by default.

Run a small smoke test:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow `
  examples/Julia_analyses/UQ_workflow/applications/texas_offshore_field/run_texas_field_sampling.jl `
  --max-geologies 1 --case-ids 1,2 `
  --output-root D:/codex_gom/UQ_workflow/texas_offshore_field_sampling_smoke
```

Run the full application:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow `
  examples/Julia_analyses/UQ_workflow/applications/texas_offshore_field/run_texas_field_sampling.jl
```

Main outputs:

- `texas_field_unique_draw_values.csv`: one row per
  geology/case/draw-group/window.
- `texas_field_slice_window_values.csv`: one row per
  geology/case/slice/window.
- `texas_field_slice_case_matrix.csv`: one row per geology/case/slice, with
  all six window permeability vectors in wide format.
- `texas_field_slice_draw_groups.csv`: mapping from field slices to draw
  groups.
- `texas_field_sampling_compact.mat`: compact MATLAB/MRST-ready arrays
  generated from the CSV outputs.
- `texas_field_sampling_report.txt`: compact run summary.

The MAT file stores `fieldPerm.logK` and `fieldPerm.perm` with dimensions:

```text
geology x level3_case x slice x window x component
```

The component order is `kxx`, `kyy`, `kzz`. Replay-oriented fields such as
`selected_sample_index`, `source_seed_base`, `exact_replay_seed`, and
`source_checkpoint_file` are included so selected realizations can be replayed
later for fine-scale properties.
