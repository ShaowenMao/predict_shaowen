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

The dynamic Kr model adds an artificial sand layer at its flow boundary. For
ordinary replay maps, that layer retains the original implementation's mean
properties over realized fault-core sand cells. A valid all-smear replay has
no such cells; in that edge case only, the boundary properties are rebuilt
from the replayed parent sand units using thickness-weighted porosity and
rotated permeability tensors. This fallback does not alter the replayed fault
map and prevents undefined pore volumes in the numerical boundary layer.

The 1D Corey history match uses the effective irreducible-water endpoint
computed by the invasion-percolation Pc curve. Native Pc discretization can
place this upscaled endpoint slightly outside the two constituent reference
endpoints. The runtime therefore validates the physical bound
`0 <= effective Swi < 1`, reports constituent-range excursions, and retains
the exact Pc-derived endpoint instead of clipping it. This keeps Pc and Kr
endpoints consistent.

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

If the recorded smoke job has already completed successfully, the submitter
verifies that state with Slurm and treats the gate as satisfied rather than
asking Slurm to create a dependency on an aged-out job record.

The submission script extracts one compact, hash-recorded 522-row assignment
table per case, writes all outputs beneath a deterministic scratch run root,
and records every Slurm dependency in `submission_manifest.csv`. To continue
an interrupted batch without deleting valid replay, Pc, or Kr checkpoints,
resubmit with the same `BATCH_ID` and `RESUME=1`.

The qualification default for exact-replay verification is `1e-3` in
`log10(k)`, equivalent to about 0.23% relative permeability. This tolerance
admits small cross-platform differences in the effective-permeability linear
solve while remaining far too tight to accept a different stochastic
realization. Override it with `REPLAY_TOLERANCE_LOG10` only when the reason is
documented in the batch provenance.

Summarize jobs and stage completion markers with:

```bash
bash summarize_qualification_batch.sh <batch_root>
```

If replay and Pc are complete but Kr must be resumed with a corrected frozen
source, submit only the missing Kr stages with:

```bash
FREEZE_ROOT=<verified_freeze_root> \
BATCH_ROOT=<existing_batch_root> \
SBATCH_EXCLUDE_NODES=node3312 \
  bash submit_qualification_kr_resume.sh
```

The resume script requires both upstream completion markers, skips existing
Kr completion markers, requires AMGCL, and writes a separate resubmission
manifest containing each Slurm job ID, frozen commit, freeze root, and stage
script hash.

# Node-bundled checkpoint execution

Large checkpoint replay/Pc campaigns can use
`submit_checkpoint_node_bundle.sh` instead of one high-memory Slurm allocation
per checkpoint group. The node-bundled path changes only scheduling: it calls
the same immutable `run_checkpoint_replay_pc.sh` scientific worker and writes
the same validated checkpoint outputs.

The scheduler first builds a deterministic lane manifest from checkpoint groups
that lack a current completion marker. Groups are greedily balanced by replay
task count. Each lane runs serially on one CPU, while several independent lanes
share a node allocation and node-local `/tmp` storage. A bundle-level preflight
rejects nodes without the configured local-space allowance. Completion remains
restartable at checkpoint-group granularity.

Recommended rollout:

1. Run `plan pilot`, then `submit pilot` with one node and 12 lanes. The pilot
   selects the 12 largest unfinished groups as a memory and storage stress test.
2. Validate `checkpoint_bundle_completion.json`, Slurm `MaxRSS`, and node-local
   storage use.
3. Run `plan full`, then `submit full`. The default full allocation uses six
   nodes with 12 one-core lanes per node and 256 GiB per node.

The scientific workflow checkout and PREDICT physics checkout are verified
against `phase_run_identity.json`; scheduler provenance is recorded separately
in `checkpoint_bundle_submission.json`.

For a complete production chain, submit the full checkpoint bundle first and
pass its numeric Slurm job ID to `submit_independent_full_fault_phase.sh` as
`EXTERNAL_CHECKPOINT_JOB_ID`. The phase launcher does not create a duplicate
checkpoint array; it attaches the standard validation gate and unchanged
assembly, dynamic-Kr, and final-QA stages to the external bundle.
