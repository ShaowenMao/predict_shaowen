# Pc100 / Kr10: Complete Fault and Bounded Dispatcher Qualification

Completed on Engaging, 2026-09-16. Reference fault: `s03_c022_case12` (task 1137).

## Decision

PASS for the complete 522-placement fault assembly, native importer, corrected-
hysteresis full-grid preflight, and bounded dispatcher paths exercised here.
This is not a full 2430-case launch, a reservoir time-marching validation, or a
claim that ten-state Kr interpolation reproduces full100 reservoir QoIs.

Production code/results/jobs and the old paused upscaling campaign were untouched.
The finite controller exited; recurring automatic checks remain off.

## Scientific and Assembly Checks

- 522 placements resolve to 510 exact scientific input identities. Reuse required
  identical fine-grid geometry/properties and scientific settings, not similar
  endpoints or average permeability.
- All 510 accepted archived full100 curves were reused. Sparse derivation retained
  every original Pc sample and selected Kr value exactly. No full100 MIP or flow
  calculations were repeated for those owners.
- All 522 Python selections exactly match the qualified Julia selector: ten
  ordered existing states nearest uniform bulk-Sg targets under the locked
  assignment rule, retaining both endpoints.
- WRR60 fine curves, cell-specific Pc, outlet-connected `krw > 0` trapping,
  physical pre-breakthrough saturation, endpoints and corrected dip-parallel
  direct flow solver are unchanged. Only Kr sampling differs from full100.
- All 522 placements passed MATLAB assembly, table and specific-file round trips.
  Original mapping, geometry, intrinsic permeability and porosity are preserved.
- Export needed zero Kr cleanup. The existing duplicate/endpoint policy produces
  97-99 SGOF rows; the underlying raw Pc100 data remain separately preserved.

## Runtime Checks

- 5,000,748 native value/derivative assertions plus 14 endpoint regressions passed
  under both Julia 1.12.6 and production Julia 1.10.4, over 624,818 evaluation
  locations across 522 curves.
- Maximum native interpolation discrepancy: Kr 2.332e-15, Pc 3.726e-9 Pa.
  These are import discrepancies, not ten-versus-100 sampling errors.
- The previously qualified endpoint oracle explains a tiny nonzero native Krg
  at Sg=0 for 407 placements using the pinned importer's 1e-16 knot shift. Maximum
  value is 5.036e-16. Raw zero nodes, importer code and the original 1e-9 overall
  import-accuracy limit were not changed.
- Full-grid preflight passed: 2,165,082 cells, including 32,190 fault cells and
  522 fault regions. All imported fault tables matched exactly; initial pressure
  discrepancy was zero and permeability tensors were positive definite.
- Host hysteresis matched the immutable oracle, including history and accounting
  witnesses. Fault mobility/QoI paths are drainage-only. Hysteresis verification
  passed 19,669 fault tests and 272 host tests without changing drainage curves.
- The preflight used `do_sim=false`; it did not advance physical simulation time.
- Production Julia emitted HostCPUFeatures cached-feature warnings. The native
  and full-grid checks passed on the tested node; general cross-node portability
  of that legacy reservoir cache is not certified here. The isolated upscaler's
  strict existing generic warm cache passed its own launcher load test.

## Dispatcher and Resources

- All 18 dispatcher tests passed on Engaging, including single-writer exclusion,
  resource admission and uncertain-submission handling.
- Live refill, exact-owner read-through, dependency gates and controlled partial-
  batch restart passed. The restarted first batch reused its two completed owners.
- A fresh four-record warm numerical canary passed, including repeat-core checks.
  Its 52.3 seconds is a canary timing, not a fresh 522-core timing estimate.
- Completed-dispatch re-entry submitted nothing and changed no submission intents.
- Every successful stage's Slurm accounting confirms COMPLETED, exit 0, advanced
  account/QoS, and the stage-specific CPU/memory request.
- Admission uses a 500-CPU / 1984-GiB planning ceiling and counts own pending work
  plus running/eligible external work. Blocked external dependency/held jobs are
  logged rather than counted as simultaneously runnable. Slurm is the final
  account-resource arbiter; this snapshot cannot reserve capacity against other
  independently submitting controllers.
- Use `qualified_dispatch.py` and `QUALIFIED_LAUNCHER_LOCK.json` going forward.
  This corrects only launcher selection/environment around the frozen science.

| Stage | Observed whole-job peak | Reservation | Successful job |
| --- | ---: | ---: | --- |
| Streaming identity audit | 45.1 MiB | 1024 MiB | 22865827 |
| Sparse derivation batches | 53.0-119.7 MiB | 512 MiB each | 16 batches |
| Warm four-record flow canary | 1051.3 MiB | 2816 MiB | 22865879 |
| Full-fault assembly/export | 2270.9 MiB | 4096 MiB | 22866046 |
| Native import, Julia 1.12 | 2242.9 MiB | 3072 MiB | 22866074 |
| Native + full-grid, Julia 1.10 | 10495.1 MiB | 12416 MiB | 22866278 |

All successful stages had zero memory-limit events. These are different stages,
not one blanket per-core allocation. Existing measured larger-class worker
qualification remains separate; this reference-fault reuse run does not provide
a fresh numerical peak for every core class. Retain the measured relative-
headroom policy rather than copying the full-grid memory request to core jobs.

## Sampling Errors and Scope

448 of 510 unique owners are flagged by the existing `max absolute Kr error >
0.08` review criterion, corresponding to 459 of 522 placements. Across this fault,
maximum absolute error against full100 is 0.205848 for gas and 0.626545 for brine.
These are budgeted sampling approximations, not failed flow solves. They are
retained in `PLACEMENT_REVIEW.csv`; no error was hidden by smoothing or extra
states. The pass establishes faithful implementation, export and runtime use of
the ten-point approximation, not sufficiently accurate leakage predictions.

## Repairs and Remaining Launch Work

The earlier audit's streaming-cache/resource issue was isolated in v1. v2 keeps
file hashes and semantic flow-log hashes distinct and verifies reader equivalence.
Two v2 launcher problems were repaired only at their failed stage: short MATLAB
temporary/preferences directories were restored, and the reservoir preflight was
returned to production Julia 1.10.4 instead of upscaler Julia 1.12.6. Failed logs
and successful numerical artifacts remain intact. Consolidated launcher probes
22866395/22866397 and zero-solve warm-cache probe 22866505 passed.

Before campaign expansion, reconcile the 2430-case owner manifest and paused
ledger, wire the new namespace to the qualified dispatcher, and admit only the
measured/qualified worker resource classes. This is campaign bookkeeping and
controlled expansion, not a reason to rerun this fault's successful upscaling.
No full campaign was launched in this qualification.

## Evidence

- `COMPLETE_REVIEW.json`: final verification, exact evidence hashes and accounting.
- `QUALIFICATION.json`: finite DAG completion and stage receipt hashes.
- `PLACEMENT_REVIEW.csv`: per-placement owner identities and sampling errors.
- `receipts/`, `intents/`, `tests.json`: immutable stage/dispatch evidence.
- `native_production/constitutive_pass.toml`, `native/constitutive_pass.toml`:
  native checks in both Julia runtimes.
- `fullgrid/PASS.toml`, `fullgrid/fault_hysteresis_validation.toml`: full-grid checks.
- Remote root: `/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/wrr60_pc100_kr10_campaign_20260916_v1/full_fault_dispatch_v2`.
