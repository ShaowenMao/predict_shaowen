# UQ Workflow

This folder contains the new Julia implementation of the updated UQ workflow.
The older exploratory scripts in
`examples/Julia_analyses/UQ_for_smart_sampling/` are intentionally left
untouched.

Current implementation focus:

1. `level1/`: placeholder for geology design and PREDICT data generation.
2. `level2/`: empirical within-window state construction.
3. `level3/`: placeholder for cross-window coupling uncertainty.

For now, Level 2 is developed against a proxy dataset:

- one provisional geology id: `g_ref`
- six fixed windows: `famp1` to `famp6`
- one 2000-run library per window:
  `examples/gom_reference_floor_full/data/<window>/small_runs/N2000_repeat01.mat`

The other `N2000_repeatXX.mat` files are reserved as holdout repeats for
Level 2 validation.

## Folder structure

- `configs/`: workflow constants and proxy manifests
- `level1/`: placeholder notes for future geology-level inputs
- `level2/lib/`: shared Julia helpers for Level 2
- `level2/workflow/`: modular Level 2 driver, config, and step files
- `level2/scripts/`: executable review, export, sampling, and validation
  utilities used by the modular driver
- `level2/outputs/`: default in-repo output location
- `level3/`: placeholder notes for future coupling scripts

## Recommended entry points

- Run the full modular Level 2 workflow:
  `level2/workflow/run_level2_workflow.jl`
- Configure Level 2 run toggles and user-facing parameters:
  `level2/workflow/level2_workflow_config.toml`
- Validate against holdout repeats:
  `level2/scripts/90_validation/validate_holdout_repeats.jl`
