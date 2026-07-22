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

## Qualification batch

`qualification_cases.csv` defines a controlled 24-case qualification set:
one `c012` geology from each of the six thickness scenarios, crossed with
Level 3 cases 01, 03, 04, and 07. The non-thickness geologic controls are held
fixed so differences across scenarios isolate the thickness architecture.

Each case runs as a restartable three-stage Slurm chain:

```text
exact PREDICT replay -> invasion-percolation Pc -> dynamic Kr (Swi medoid)
```

The smoke replay/Pc stages use one slice across all six windows. Its Kr stage
uses one representative curve on the complete production-size 3D grid; it
does not use the artificial cropped-grid plumbing mode.

First submit the smoke gate:

```bash
bash submit_qualification_batch.sh smoke
```

Then use the printed Kr job ID to hold the full batch until the smoke chain
finishes successfully:

```bash
QUALIFICATION_GATE_JOB_ID=<smoke_kr_job_id> \
  bash submit_qualification_batch.sh full
```

The submission script extracts one compact, hash-recorded 522-row assignment
table per case, writes all outputs beneath a deterministic scratch run root,
and records every Slurm dependency in `submission_manifest.csv`. To continue
an interrupted batch without deleting valid replay, Pc, or Kr checkpoints,
resubmit with the same `BATCH_ID` and `RESUME=1`.

Summarize jobs and stage completion markers with:

```bash
bash summarize_qualification_batch.sh <batch_root>
```
