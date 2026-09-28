# Full-fault Julia qualification and paired production comparison

This is a separate candidate study. No production code, data, jobs, scheduling,
or reservoir simulation is changed. No reservoir rerun is submitted here.

## Case And Controls

Use completed production case s03_c022_case12, the intermediate-leakage
representative: high sand proportion, uniform parent-layer thickness, faulting
depth 1000 m, sand Vcl 0.2 and clay Vcl 0.4. Its original reservoir simulation
has saved 1000-year results and prior pressure, CO2 and brine analyses.

The production property completion record covers 522 Pc assignments and six
dynamic-Kr representatives. Select those SAME six representatives as pilots,
plus the globally minimum- and maximum-Kzz assignments. Cover the remaining
assignments only after all pilots pass numerical and measured-storage gates.

All 6 x 87 assignments retain their exact window, slice, checkpoint SHA256,
sample index, accepted replay seed, task identity and PREDICT revision.
The 522 assignments contain 510 distinct replay identities. This first rollout
computes all 522 independently; it does not reuse curves across assignments.
The previously validated within-core byte-identical mobility cache is retained.

The original production native Pc, six dynamically fitted Kr shapes and
slice-mapped Kr curves are immutable comparison baselines, NOT ground truth.
Expected differences combine cell-based versus regional Pc scaling, trapping,
the sand endpoint, physical pre-breakthrough saturation, and dynamic versus
MIP-based Kr. This full-workflow comparison cannot attribute effects uniquely.

## Computation

1. Freeze the exact case assignment map, production curve/porosity summaries,
   completed property MAT and source receipts. Verify all 522 seed/sample
   identities against the existing reservoir-ready property artifact.
2. Replay the frozen cores using the existing PREDICT/MRST geometry workflow.
   The stochastic identity is exact; intrinsic replay tolerance remains 0.005
   log10 units. Check grid count and physical pore volume against production.
   No new stochastic samples or geometry simplification are allowed.
3. Export geometry-only TPFA half-face transmissibilities, physical fine-grid
   properties and the unchanged two reference tables. Julia scales Pc per
   gridblock, uses persistent outlet-connected krw > 0 trapping with sand
   endpoint 0.95, and solves dip-parallel Kr with the validated corrected-direct
   solver and unchanged binary128 helper.
4. Use fixed 100 output pressures. A deterministic pressure-only dictionary
   combines the original production pressure nodes, 0, 0.98 times the smallest
   local entry pressure, 65 log-spaced nodes and the largest local endpoint.
   Two geometric-midpoint subdivisions precede nearest-log fixed100 selection.
   Dictionary construction does not run extra MIP states or phase-flow solves.
   Every mandatory entry/zero-mobility event is still processed internally.
5. Recompute intrinsic conductance on the identical network for normalization.
   Compare resulting Kzz with the canonical production value without changing
   that production value. Both phases and Pc share the same physical MIP state.

The validated integrated numerical sources are vendored byte-for-byte. New
code handles raw input packaging, pressure-only planning, scheduling, restart,
comparison and export audits; it does not replace the MIP/phase-flow kernels.
The kernel source SHA256 is
d6de18a5bb2428fde69846211c51e7fc535f28793c62d3b1e2985d4f88d46a8a.

## Reservoir Export Gates

Preserve all 100 native rows with physical bulk Sg, Pc in Pa, both Kr, trapping,
local-immobility, gas-breakthrough and invaded-PV diagnostics. Check finite
values, bounds, pressure ordering, saturation ordering, repeated/near-repeated
Sg, both Kr monotonicities, zero-flow behavior, endpoints and 522-slot coverage.
Do not clip, smooth, force a terminal Krw of zero, call the terminal Sg a
physical residual saturation, or merge pressure branches silently.

New-core interpolation errors have NOT been independently measured; fixed100
does not certify a continuous error bound. Export flags are not numerical job
failures. They are retained for review, as requested for bounded-cost sampling.

At completion, assemble a lossless candidate_raw_s03_c022_case12.mat package
and read it back to verify every row, slot identity, canonical K map, original
porosity map and coordinate contract. Deliberately use a separate candidateRaw
schema, NOT the production reservoirReady variable/schema. A scalar Pc(Sg)
table can require an explicit branch/endpoint policy; the package must not
silently pass as a production-compatible file before that policy and the
actual reservoir importer have been qualified. Production promotion and
reservoir_export_approved remain false.

Compare the native candidate and original production curves at common physical
Sg values without extrapolation. Use log-Pc interpolation only on unambiguous
positive-pressure intervals and linear Kr interpolation. Keep unavailable
comparisons explicit. Record differences, not purported accuracy errors.
Future actual-importer checks must separately assess its entry-plateau rule,
units, axis rotation, saturation mapping, hysteresis and endpoint continuation.

## Restart And Scheduling

Each stage uses a sealed source and new attempt directory in project storage.
Completed cores are reused only after their complete calculation identity and
output hashes match. Interrupted preparation is never reused. A fully recorded
preparation can feed a fresh numerical attempt for the same immutable identity.
Within-curve restart is not introduced; an interrupted numerical sweep restarts
that core. Exclusive per-slot directory leases prevent concurrent writers.
After a hard-killed job, a stale lease requires explicit scheduler verification
before removal; it is never silently stolen.

Initialization, pilot array, storage/numerical gate, remaining-core array and
final assembly use afterok dependencies, no aftercorr wildcard and no array
concurrency cap. Each task is nonexclusive with 2 CPUs and 18 GB. Existing
production resource reservations and dependencies are untouched. Heavy work
runs on compute nodes; shared scratch receives no dense results. Node-local
/tmp holds only disposable runtime caches/preferences. Dense MIP state archives
are disabled for the full-fault curves. Replayed geometry and raw inputs remain
in project storage for traceability; the pilot gate measures their actual size
before releasing the full array.

## Remaining Decisions

Successful numerical completion is not production adoption. Inspect paired
property maps/curves, resolve export-policy flags without hiding method changes,
qualify the actual reservoir import and bounded canary setup, and assess broader
geologic coverage before changing any production campaign.
