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

## Intentionally deferred

- Phase 2 geology classification requires Phase 1 reservoir QoIs and scientific
  thresholds, so the selector is implemented but no 12-geology list is frozen.
- The optional clustered/fragmented along-strike permutation experiment is not
  part of the primary production design and remains disabled.
- The full 162-geology production manifest is generated only from a clean,
  committed isolated branch. The scientific implementation and one-geology
  end-to-end acceptance are complete; production freezing is intentionally
  deferred until this version is reviewed and committed.
