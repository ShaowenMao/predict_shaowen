# `independent_full_fault_v1` Implementation Status

## Implemented and tested

- Strict loading of 2,000 accepted joint PREDICT realizations per window.
- Exact full-, low-, and high-library medoids in physical 3D log10(k) space.
- Stable SHA-256 counter seeds and uniform sampling with replacement.
- Complete Phase 1 and Phase 2 case manifests with 6 x 87 coverage.
- Explicit probabilistic, benchmark, and stress roles.
- Marginal-fidelity and accidental-coupling QC with bootstrap references.
- Atomic CSV/MAT/replay-Pc handoff with restartable unique task identities.
- Exact checkpoint, realization, seed, commit, configuration, and hash provenance.
- Compact fault-local permeability MAT and full CSV/MAT round-trip validation.
- Role/provenance propagation through Pc assembly and reservoir-ready export.
- Explicit fault-local to reservoir-grid tensor contract and MATLAB reference tests.
- Role-safe Phase 1 QoI summaries and corrected uniform-design variance decomposition.
- Configurable, deterministic category-stratified Phase 2 maximin selection.
- Hard completeness gates for the 162-geology Phase 1 rectangle and the
  12-geology Phase 2 selection.
- Current-marker validation that rejects stale revised-case completion markers.

## Runtime-qualified

- One real geology produces all 15 Phase 1 cases and 7,830 assignments.
- The real handoff deduplicates 7,830 assignments into 4,902 replay/Pc tasks.
- Exact replay reproduces all six deterministic representative medoids within
  the existing permeability tolerance.
- Rigorous native-endpoint invasion-percolation Pc completes for all six
  deterministic representative windows.
- Engaging job `19980738` completes exact replay, all six Pc outputs, six
  full-3D dynamic-Kr representatives with required AMGCL, 522 full-slice
  assignments, and final MAT export in 31 min 59 s using six CPUs and 48 GB.
- Strict post-finalization verification reports zero errors and zero source
  log-permeability mismatches in `acceptance_verification_v2.json`.
- Engaging checkpoint-bundle pilot `20001840` completes the 12 largest
  unfinished replay/Pc checkpoint groups with 12 one-core lanes in 2 h 54 min
  47 s. All 12 completion markers pass identity and hash validation. The
  largest observed lane RSS is approximately 16.3 GiB, supporting an 18 GiB
  shared-memory budget per replay/Pc lane. This budget does not replace the
  separately qualified 48 GiB dynamic-Kr allocation.

## Phase 1 production launch

- Production run `independent_full_fault_v1_phase1_20260809_v1` resumes 34
  validated checkpoint groups and schedules the remaining 938 groups without
  recomputing completed work.
- High-memory node array `20012444` and its pending downstream chain are
  superseded by standard checkpoint-chunk array `20028083`. The replacement
  restores the proven production architecture: 195 restartable tasks, five
  checkpoint groups per task, one CPU and 18 GiB per task, up to 96 concurrent
  tasks, shared flash scratch, and no exclusive-node request.
- Slurm `--exclusive=user` does not isolate jobs owned by the same user; it
  excludes other users and unnecessarily reduces eligible nodes. Optional
  node-local schedulers now require true whole-node exclusivity and are
  documented as diagnostic paths rather than the production default.
- Validation gate `20028084`, geology assembly array `20028085`, dynamic-Kr
  array `20028086`, and final QA gate `20028087` form the replacement dependency
  chain. The transition preserves all 34 validated checkpoint outputs and the
  immutable workflow, PREDICT-physics, method-config, sampling, and seed
  identities.
- Submission metadata, immutable commit identifiers, method hashes, lane
  manifests, and completion markers are stored beneath the permanent run root
  on `/orcd/data/juanes/001/shaowen`.

## Intentionally deferred

- Phase 2 geology classification requires Phase 1 reservoir QoIs and scientific
  thresholds, so the selector is implemented but no 12-geology list is frozen.
- The optional clustered/fragmented along-strike permutation experiment is not
  part of the primary production design and remains disabled.
