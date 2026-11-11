# Phase 1 Pc/Kr Upscaling Workflow

Reusable **reference implementation and operating guide**, preserved from the
completed Phase 1 campaign. Start here for Phase 2, Phase 3, or a new campaign.
This is not a new portable launcher, and checking it out does not authorize jobs.

## What Finished

The saved completion audit dated **2026-11-11 15:52 UTC** records **2,592/2,592
cases and 1,353,024/1,353,024 core positions**, with no remaining exceptions.
Each case contains six windows of 87 strike positions (522 cores).
See [the original audit](reference/phase1_completion_audit_20261111_v1/COMPLETION_AUDIT.json).
That audit checked canonical case receipts and plan seals, not a fresh rehash of
every old archive. It is historical evidence, not a live cluster status check.

The scientific configuration is 324 scenarios x eight independent stochastic
realizations, samples 4 and 6 for both host-clay and fault-clay master Pc at
40 degrees, 100 Pc states, and the qualified adaptive selector using at most
ten shared gas/brine Kr states. Sample-4 and sample-6 realizations are independent,
not identical-geometry pairs. No old campaign results were relabeled.

## Start Here

From the repository root, with Python 3.12 or newer:

```powershell
python -B utils/phase1_upscaling/verify_package.py
python -B utils/phase1_upscaling/run_offline_tests.py
```

Both commands are local. They do not connect to Engaging, submit jobs, import a
production worker entrypoint, or solve a physical flow system. The test suites
import worker modules only for mocked, temporary-directory regression tests.
Linux process-group tests are explicitly skipped on Windows. Additional
export tests need NumPy; see [validation](VALIDATION.md).

| Guide | Purpose |
| --- | --- |
| [Operations](OPERATIONS.md) | Scheduling, checkpoints, recovery, results and storage |
| [Next campaign](NEXT_CAMPAIGN.md) | Minimal adaptation and qualification checklist |
| [Provenance](PROVENANCE.md) | Source versions, dependencies, boundaries and evidence |
| [Validation](VALIDATION.md) | What the repository checks establish and do not establish |

## Source Map

`reference/` preserves original file bytes, source LOCKs, tests and selected
handoffs. Names retain dates/versions so predecessor tests remain meaningful.
`SOURCE_MANIFEST.json` pins every included file and explicitly enumerates omitted
bulk configuration files. `science/` is a readable, checksum-matched mirror of
the production v5 scientific ZIP, **not** a separately qualified engine.

| Original package under `reference/` | Role |
| --- | --- |
| `phase1_pc40_adaptive10_2592_configuration_20261030_v2` | Scientific configuration, catalog, profiles, adaptive selector, sampling source |
| `phase1_pc40_adaptive10_runner_20261030_v4` | Qualified cache-runner predecessor; used by transition regression tests |
| `phase1_pc40_adaptive10_runner_20261030_v5` | Actual first-pass runner, finite handoffs and source-pinned transition |
| `phase1_round2_20261109_v4` | Allocation-aware timeout/finalization and wider recovery-only window |
| `phase1_round3_20261110_v1` | One-core/four-hour recovery and exact saved-prefix handling |
| `phase1_exact_spacing_recovery_20261110_v1` | Tested strict-positive saturation-spacing export rule |
| `phase1_saved16_closeout_20261110_v2` | Bounded publication of the 16 saved-flow export exceptions |
| `phase1_solver_targeted_20261110_v1` | Full recovery after a strict failed-state solver probe |
| `phase1_solver_window8_20261110_v2` | Qualified overlapping targeted batches; 64 outstanding workers |
| `phase1_precision_recovery_20261111_v1` | High-precision audit and saved-call reuse |
| `phase1_precision_batches_20261111_v1` | Bounded precision-recovery batches and publication |
| `phase1_direct_primary8_20261111_v1` | Direct-primary recovery predecessor, retained for numerical-identity regression |
| `phase1_final_reference4_20261111_v2` | Saved-primary/direct-reference recovery, sparse direct dependencies and accounting fix |
| `phase1_completion_audit_20261111_v1` | Final read-only catalog/receipt audit and evidence |

## Important Boundary

The historical scripts deliberately retain their **original absolute cluster
paths, case scope, pins and launch gates**. Do not execute their production,
transition, seal, generation or publication CLIs against a new campaign without
adaptation. Do not rerun a historical seed, reset a cursor, remove an intent, or
point a new campaign at Phase 1 output. The original campaign remains complete;
the older 2,430-case campaign remains paused.

This commit adds documentation and source snapshots only. It neither adopts a
new runner nor merges the later exception solvers into normal production.
