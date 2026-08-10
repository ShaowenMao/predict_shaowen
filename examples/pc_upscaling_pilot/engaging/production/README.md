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

The qualification default for replay numerical equivalence is `0.005` in
`log10(k)`, equivalent to about 1.16% relative permeability. This tolerance
admits small cross-platform differences in the effective-permeability linear
solve while remaining far too tight to accept a different stochastic
realization. The stochastic identity checks remain exact. Override it with
`REPLAY_TOLERANCE_LOG10` only when the reason is documented in the batch
provenance.

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

# Standard checkpoint production execution

Production replay/Pc campaigns use the restartable checkpoint-chunk array in
`submit_checkpoint_replay_pc.sh`, or the equivalent checkpoint stage in
`submit_independent_full_fault_phase.sh`. Each Slurm task uses one CPU and an
18 GiB memory request, processes five checkpoint groups serially by default,
and skips groups that already have a valid completion marker. Up to 96 tasks
may run concurrently. Temporary replay, Pc, and MATLAB runtime data use each
allocated node's local `/tmp`; shared flash scratch is reserved for Slurm logs.
The tasks do not require exclusive nodes.

This is the default production architecture. It matches the successful
1,620-case campaign and keeps scheduling independent from the scientific
worker and output contract.

Production replay uses `0.005` log10 units as the validated maximum
permeability numerical-equivalence difference across MATLAB/MRST platforms.
This tolerance applies only to the continuous effective-permeability solve.
The selected library row, checkpoint-recorded accepted seed and attempt,
checkpoint SHA-256, PREDICT physics commit, and method-configuration SHA-256
must still match exactly. Newly replayed groups also record a canonical
SHA-256 of the reconstructed integer material-unit map. Completion gates
accept older successful markers produced with a stricter tolerance, so a
continuation submits only missing or invalid groups.

The independent-full-fault Phase 1 architecture covers 972 groups with 195
tasks, five groups per task, one CPU and 18 GiB per task, a 24-hour wall time,
and a 96-task concurrency cap. A production incident showed that the legacy
shared-scratch temporary default exceeds the flash quota at this concurrency.
The node-local default is therefore part of the scheduler contract and is
covered by regression tests; it does not change completed outputs or scientific
provenance.

# Optional node-bundled checkpoint diagnostics

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

Diagnostic rollout:

1. Run `plan pilot`, then `submit pilot` with one node and 12 lanes. The pilot
   selects the 12 largest unfinished groups as a memory and storage stress test.
2. Validate `checkpoint_bundle_completion.json`, Slurm `MaxRSS`, and node-local
   storage use.
3. Use a full node-bundled campaign only when there is a demonstrated reason
   to prefer node-local storage over the standard checkpoint-chunk array.

The production memory basis is Engaging job `20001840`, which runs the 12
largest unfinished checkpoint groups concurrently and completes all groups in
2 h 54 min 47 s. Slurm reports a maximum lane RSS of about 16.3 GiB. Because
memory is requested once per node, the 216 GiB allocation is shared flexibly
across the 12 lanes rather than imposing an independent 18 GiB hard limit on
each MATLAB process.

The scientific workflow checkout and PREDICT physics checkout are verified
against `phase_run_identity.json`; scheduler provenance is recorded separately
in `checkpoint_bundle_submission.json`.

## Optional one-node array diagnostic

`submit_checkpoint_node_array.sh` uses the same deterministic lane manifest and
scientific worker but submits one node per Slurm array task. Each node runs 12
one-core lanes with the same 216 GiB shared-memory request. This layout lets
individual nodes backfill and start independently, reducing the scheduling
penalty of requiring a multi-node allocation to start atomically. Array-task
completion is validated over its disjoint global lane-ID range, and the final
checkpoint gate still validates every checkpoint group before case assembly.
The default uses true whole-node `--exclusive` allocation because deterministic
node-local `/tmp` accounting requires a dedicated node. Slurm
`--exclusive=user` is not sufficient: it excludes other users but can still
co-locate jobs from the same user. Because whole-node requests can wait much
longer in the queue, this path is diagnostic rather than the production
default.

For a safe scheduling comparison with a pending node-bundle job, pass its job
ID as `FALLBACK_JOB_ID`. The array launcher holds that fallback before it
submits the array, preventing concurrent writes to the same checkpoint output.
If the array completes, cancel the held fallback to release an existing
`afterany` validation dependency. If the array is abandoned, cancel every
array task before releasing the fallback. Submission provenance and this
handoff protocol are recorded in `checkpoint_node_array_submission.json`.

For a complete production chain, submit the full checkpoint bundle first and
pass its numeric Slurm job ID to `submit_independent_full_fault_phase.sh` as
`EXTERNAL_CHECKPOINT_JOB_ID`. The phase launcher does not create a duplicate
checkpoint array; it attaches the standard validation gate and unchanged
assembly, dynamic-Kr, and final-QA stages to the external bundle.
