# Production Freeze and Restartable Manifest

This folder freezes the updated collapsed-layer, `cell_union_psmear` inputs
before full Engaging replay, invasion-percolation Pc upscaling, and dynamic Kr
upscaling.

The manifest deduplicates repeated field assignments. A replay/Pc task is
identified by:

- PREDICT checkpoint relative path and SHA-256;
- selected realization index;
- exact replay seed;
- committed source revision; and
- SHA-256 of `production_method_config.toml`.

Changing any of these fields creates a different task ID, preventing stale
replay or Pc artifacts from being reused.

## Build

Run from the repository root on the Windows workstation:

```powershell
python examples/pc_upscaling_pilot/engaging/production/build_restartable_manifest.py `
  --sampling-root D:/codex_gom/UQ_workflow/texas_offshore_field_sampling_collapsed_cell_union `
  --predict-root D:/Github/predict_shaowen/examples/thickness_scenario_data_collapsed_cell_union `
  --method-config examples/pc_upscaling_pilot/engaging/production/production_method_config.toml `
  --repo-root D:/Github/predict_shaowen `
  --output-root D:/codex_gom/UQ_workflow/production_freeze_collapsed_cell_union_20260722_v1 `
  --remote-freeze-root /orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v1
```

## Restart semantics

The manifest records deterministic output and completion-marker paths. A
stage is reusable only when both files exist and the marker matches all task
provenance fields. Missing, failed, or mismatched tasks are rerun individually;
completed valid tasks are not recomputed.

`assignment_to_task.csv` maps every geology/case/slice/window assignment back
to its content-addressed task. `task_shards/` divides the unique workload into
bounded files suitable for Slurm arrays.

## Verify

Verify generated manifests locally:

```powershell
python examples/pc_upscaling_pilot/engaging/production/verify_restartable_manifest.py `
  --manifest-root D:/codex_gom/UQ_workflow/production_freeze_collapsed_cell_union_20260722_v1
```

On Engaging, also provide the copied input roots to validate every transferred
file against its SHA-256 inventory.
