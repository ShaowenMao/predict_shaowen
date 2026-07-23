# Engaging Production Workflow

This workflow runs exact PREDICT replay, native-endpoint invasion-percolation
Pc upscaling, and robust dynamic Kr upscaling for the 162-geology,
10-case-per-geology Texas offshore ensemble.

## Scientific Invariants

- Frozen PREDICT input: collapsed adjacent lithology.
- Smear overlap rule: `cell_union_psmear`.
- PREDICT library size: 2,000 realizations per geology-window checkpoint.
- Field coverage: 162 geologies, 10 cases, 87 slices, and six windows.
- Fine replay and dynamic Kr retain the full strike dimension.
- Pc uses invasion percolation with native endpoints and effective Swi.
- Kr uses one effective-Swi medoid realization per case-window.
- Dynamic Kr requires AMGCL; no silent linear-solver fallback is allowed.
- Reservoir exports retain both `full_slice` and `pe_branch_medoid` Pc options.

The semantic preflight refuses to run when any of these conditions is false.

## Storage Layout

Durable, compact results are stored under:

```text
/orcd/data/juanes/001/shaowen/predict_shaowen/production_runs/<run_id>/
```

Large temporary replay maps are written to node-local temporary storage when
available, with scratch as the fallback. They are deleted only after replay
verification and compact Pc publication succeed.

```text
checkpoint_manifest/   one selection per geology-window checkpoint
checkpoint_pc/         compact Pc, porosity, effective-Swi, and replay provenance
case_work_manifest/    small per-geology assignment tables
case_inputs/           reconstructed 6x87 Pc tables and six Kr representatives
case_results/          validated reservoir-ready Pc/Kr MAT files
```

Slurm logs are written to flash scratch:

```text
/home/shaowen/orcd/scratch/predict_shaowen/production_logs/<run_id>/
```

## Phase 1: Checkpoint-Centered Replay and Pc

Qualification tranche:

```bash
cd /home/shaowen/orcd/pool/predict_shaowen
bash examples/pc_upscaling_pilot/engaging/production/submit_checkpoint_replay_pc.sh qualification60
```

Full 1,620-case production:

```bash
bash examples/pc_upscaling_pilot/engaging/production/submit_checkpoint_replay_pc.sh full
```

The qualification tranche contains all ten Level-3 case types for one geology
from each of the six thickness scenarios. It produces 36 checkpoint jobs.
The full run produces 972 checkpoint jobs.

Each checkpoint job:

1. replays every unique selected realization for one geology-window;
2. verifies all three permeability components against the frozen checkpoint;
3. computes native Pc, effective Swi, and volume-weighted porosity;
4. publishes compact task-addressed files atomically;
5. removes temporary fine-scale maps.

Re-running the same submission is safe. A checkpoint is skipped only when its
done marker matches the checkpoint hash, frozen physics commit, and method hash.

## Phase 2: Case Assembly and Dynamic Kr

After Phase 1 is running or complete:

```bash
bash examples/pc_upscaling_pilot/engaging/production/submit_case_assembly_kr.sh qualification60
```

For full production:

```bash
bash examples/pc_upscaling_pilot/engaging/production/submit_case_assembly_kr.sh full
```

The submission uses Slurm dependencies:

```text
checkpoint replay/Pc -> case assembly -> dynamic Kr
```

Case assembly reconstructs every 6x87 Pc curve assignment from compact
checkpoint products. It then selects the observed effective-Swi medoid for
each window. Only those six exact PREDICT realizations are replayed for
dynamic Kr.

Each dynamic-Kr job validates:

- six full-3D representative calculations;
- unchanged strike dimension;
- Pc/Swi endpoint consistency;
- monotonic and bounded Kr curves;
- complete 6x87 slice endpoint mapping;
- both reservoir-ready Pc representations.

## Restart and Failure Policy

- No tolerance is relaxed automatically.
- No failed task is silently omitted.
- An incomplete checkpoint or case directory is replaced on rerun.
- A completed output is reused only when its identity marker matches.
- Downstream jobs use `afterok` and do not run after an upstream array failure.
- Exact task provenance retains checkpoint hash, sample index, and replay seed,
  so any selected realization can be recreated later.

Use `summarize_production_status.py` to audit file-level completion:

```bash
python3 examples/pc_upscaling_pilot/engaging/production/summarize_production_status.py \
  --run-root /orcd/data/juanes/001/shaowen/predict_shaowen/production_runs/<run_id>
```

## Expansion Gates

1. One real checkpoint job must match trusted qualification results.
2. The 60-case tranche must complete all ten case types in all six thickness
   scenarios.
3. One complete 27-geology thickness scenario must pass before launching the
   remaining five scenarios.
4. Full production starts only after replay, Pc, Kr, storage, and restart
   diagnostics from the preceding gate are reviewed.
