# Independent Full-Fault Sampling v1

This directory implements `independent_full_fault_v1` without modifying the
legacy similarity-group workflow. The production generator does not call that
workflow; the legacy grouping code remains available only for diagnostics. The
new design separates probabilistic sampling from deterministic interpretation
benchmarks.

## Case design

Phase 1 generates 15 cases for each of 162 geologies:

| Case role | Count per geology | Probabilistic UQ |
| --- | ---: | :---: |
| Independent full-distribution realization | 12 | Yes |
| Full-library-medoid benchmark | 1 | No |
| Low-state-library-medoid stress case | 1 | No |
| High-state-library-medoid stress case | 1 | No |

Each case has six windows by 87 along-strike slices. Independent assignments
are uniform draws with replacement from the complete 2,000-member empirical
PREDICT library. The hash-based counter sampler is conditionally independent
across cases, windows, and slices and does not depend on parallel execution
order.

Phase 2 adds replicate IDs 13 through 52 for a reviewed list of 12 geologies.
It contains only 40 additional independent cases per selected geology.

## Directory map

```text
sampling/
  workflow/       Configuration and manifest-generation driver
  lib/            Julia library, validation, QC, and compact MAT export
  integration/    Sampling adapter, atomic handoff, and replay/Pc freeze
  production/     Engaging submission, continuation, and status tools
  analysis/       Role-safe Phase 1 analysis and Phase 2 maximin selection
```

## Generate Phase 1 manifests

Run a one-geology qualification first:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow `
  examples/Julia_analyses/UQ_workflow/sampling/workflow/run_sampling_workflow.jl `
  --phase phase1 `
  --only-geology s03_c001 `
  --output-root D:/codex_gom/UQ_workflow/independent_full_fault_v1_qualification `
  --run-ensemble-qc true `
  --qc-bootstrap-count 100
```

For production, use a clean committed worktree, remove `--only-geology`, and
set `[provenance].require_clean_code = true` in the frozen configuration.
Here, clean means that the repository has neither modified tracked files nor
untracked files.
The expected Phase 1 rectangle is 162 geologies, 2,430 cases, and 1,268,460
assignment rows.

## Sampling QC

QC is evaluated over the pooled 1,044 independent assignments per window, not
one individual full-fault case.

- Component-wise Wasserstein and joint energy distances test marginal fidelity.
- A family-wise maximum test covers all 24 distance metrics.
- Pairwise window-rank correlations diagnose accidental coupling.
- The 95% independence band remains a warning diagnostic.
- A separately configured 99% band plus exact seed-stream structure is the
  implementation-acceptance rule. This avoids rejecting about 5% of correctly
  independent geologies by construction when 162 geologies are checked.

The check verifies implementation independence; it does not prove geological
independence.

## Build the immutable property handoff

The atomic handoff command produces:

- human-readable adapted assignments and their metadata sidecar;
- a compact fault-local permeability MAT with complete role/provenance fields;
- deduplicated exact replay and Pc tasks;
- selected-checkpoint hashes, restart policy, and a full SHA-256 inventory.

```powershell
python examples/Julia_analyses/UQ_workflow/sampling/integration/build_phase_handoff.py `
  --manifest <phase_manifest.csv> `
  --geology-catalog <level1_geology_catalog.csv> `
  --predict-root <collapsed_cell_union_predict_root> `
  --repo-root <clean_checked_out_repository> `
  --output-root <immutable_phase_handoff>
```

Phase 1 and Phase 2 must be frozen separately. Production handoffs reject a
dirty worktree and a manifest/code commit mismatch. The two override flags are
for development acceptance only and are recorded in metadata.

Verify a transferred production handoff before allocating compute resources:

```bash
python3 examples/Julia_analyses/UQ_workflow/sampling/integration/verify_phase_handoff.py \
  --handoff-root "${HANDOFF_ROOT}" \
  --phase phase1
```

## Run replay, Pc, and dynamic Kr on Engaging

The phase launcher reuses the validated replay/Pc, case-assembly, dynamic-Kr,
AMGCL, and completion-gate workers. It derives all work counts and hashes from
the immutable handoff and refuses a dirty or wrong-commit checkout. It does not
call the legacy similarity-group case generator.

```bash
export HANDOFF_ROOT=/orcd/data/juanes/001/shaowen/predict_shaowen/handoffs/independent_full_fault_v1_phase1
export PREDICT_ROOT=/orcd/data/juanes/001/shaowen/predict_shaowen/inputs/thickness_scenario_data_collapsed_cell_union
export PREDICT_CODE_ROOT=/orcd/data/juanes/001/shaowen/predict_shaowen/code/predict_physics_68351e3
export RUNTIME_REPO=/orcd/data/juanes/001/shaowen/predict_shaowen/orchestration/<immutable_runtime>/runtime_repo
export ORCHESTRATION_COMMIT=<exact_runtime_commit>
export RUN_ID=independent_full_fault_v1_phase1_YYYYMMDD_v1

bash examples/Julia_analyses/UQ_workflow/sampling/production/submit_independent_full_fault_phase.sh plan phase1
bash examples/Julia_analyses/UQ_workflow/sampling/production/submit_independent_full_fault_phase.sh submit phase1
```

If walltime, node, or solver failures interrupt a run, first confirm that no
jobs with the same run ID remain active, then submit only unfinished chunks:

```bash
bash examples/Julia_analyses/UQ_workflow/sampling/production/submit_independent_full_fault_phase.sh continue phase1
```

`RUNTIME_REPO` and `PREDICT_CODE_ROOT` are deliberately separate contracts.
The runtime repository must be a clean checkout of `ORCHESTRATION_COMMIT`;
when that variable is omitted, it defaults to the frozen sampling-workflow
commit for backward compatibility. `PREDICT_CODE_ROOT` must be a clean checkout
of the historical PREDICT physics commit recorded by the source libraries. The
launcher also verifies the frozen method-configuration hash before planning or
submitting work. Exact replay resolves PREDICT functions from the physics
checkout; Pc/Kr drivers and production orchestration resolve from the runtime
checkout. A continuation may therefore use a validated orchestration fix without
changing the immutable sampling, physics, handoff, or method identities.

The status tool treats an item as complete only when its validated done marker
is present. For revised cases, the marker must also contain the current nested
assignment-provenance block and matching hashes; stale markers are rejected.
Completed cases inside a resubmitted chunk are skipped.

The validated replay numerical-equivalence tolerance is a maximum absolute
log10-permeability difference of `0.005`. Completion markers generated with a
stricter tolerance, including `0.001`, remain valid. Only missing or invalid
markers are retried; seed, realization index, checkpoint/code/configuration
hashes, and the discrete material-architecture map remain exact contracts.

```bash
python3 examples/Julia_analyses/UQ_workflow/sampling/production/phase_production_status.py \
  --run-root "${RUN_ROOT}"
```

Phase 1 defaults batch five checkpoint groups, nine geology assemblies, and 16
Kr cases per Slurm array element. The scientific work units remain separate;
batching only keeps the queued-element count below the Engaging account limit.
Large checkpoint and Kr intermediates default to node-local `/tmp`; shared
flash scratch holds only scheduler logs, while validated outputs are published
to durable project storage. This storage placement is operational only and
does not alter replay, Pc, Kr, sampling, or validation calculations.
Each Kr case uses six MATLAB process workers, and AMGCL remains required for
the 3D solve. The robust 1D setup explicitly disables CPR during solver
construction before installing its configured backslash or AMGCL solver. This
avoids loading an unused CPR class and does not change the governing model,
timesteps, Corey search, or objective function.

### Engaging acceptance

The real one-case acceptance job `19980738` completed on Engaging in 31 min
59 s using six CPUs and 48 GB. It exercised exact replay, native-endpoint
invasion-percolation Pc for all six windows, six full-3D dynamic-Kr
representatives with AMGCL, 522 full-slice assignments, and final MAT export.
No along-strike collapse was used. The durable acceptance results are stored at:

```text
/orcd/data/juanes/001/shaowen/predict_shaowen/qualification_results/independent_full_fault_v1_s03_c001_acceptance_20260808
```

The strict post-finalization report is
`acceptance_verification_v2.json` in that directory. It records complete
status, zero validation errors, and zero source log-permeability mismatches.

## Coordinate contract

The compact MAT stores the PREDICT-local diagonal components unchanged:

```text
kxx = fault-normal
kyy = along-strike
kzz = down-dip
```

They are not reservoir-grid components and are not pre-rotated. Downstream
import must use the actual paired fault-node geometry for each cell and apply
`fault_local_to_reservoir_grid_signed_yz_v1`. The MAT stores the mapping name,
equations, component meanings, unit conversions, and the requirement for a
per-cell signed angle. MATLAB reference tests verify sign, eigenvalues, and
positive definiteness.

## Analyze Phase 1 results

Reservoir results use long-form columns:

```text
geology_id,case_id,qoi_name,qoi_value
```

Run:

```powershell
python examples/Julia_analyses/UQ_workflow/sampling/analysis/analyze_phase1_results.py `
  --results-csv <phase1_qoi_results.csv> `
  --sampling-manifest <phase1_sampling_manifest_all_geologies.csv> `
  --expected-geologies 162 `
  --output-root <phase1_analysis>
```

Case roles are read from the canonical manifest. Only `independent_full` cases
enter conditional means, variances, and the corrected Level-1/Level-2
uniform-design decomposition. Medoid bias and high-minus-low stress response
are reported separately.

After scientific regime classification, select balanced Phase 2 geologies with:

```powershell
python examples/Julia_analyses/UQ_workflow/sampling/analysis/select_phase2_geologies.py `
  --candidates-csv <classified_geologies.csv> `
  --feature-columns <comma_separated_normalized_design_columns> `
  --categories <comma_separated_reviewed_categories> `
  --per-category 4 `
  --expected-total 12 `
  --output-root <phase2_selection>
```

The selector does not define leakage thresholds. It performs deterministic
category-stratified maximin coverage after classifications have been reviewed.
The production gate requires exactly three reviewed categories with four
geologies in each category.
Every source candidate column is retained in the selected table so that the
scientific category, threshold values, and review rationale remain auditable.

## Test commands

```powershell
julia --project=examples/Julia_analyses/UQ_workflow `
  examples/Julia_analyses/UQ_workflow/test/runtests.jl

python -m unittest discover `
  -s examples/Julia_analyses/UQ_workflow/sampling/integration `
  -p "test_*.py" -v

python -m unittest discover `
  -s examples/Julia_analyses/UQ_workflow/sampling/analysis `
  -p "test_*.py" -v
```
