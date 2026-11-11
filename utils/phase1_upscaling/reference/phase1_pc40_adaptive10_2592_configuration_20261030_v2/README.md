# Phase 1: 2,592 cases, 40-degree Pc and adaptive-10 Kr

This is a new, launch-disabled successor to
`phase1_pc40_2592_configuration_20261030_v1`. The parent is unchanged.
All 324 scenarios, eight independent stochastic realizations per scenario,
2,592 cases and 1,353,024 core assignments are preserved byte-for-byte.
No cases were redrawn and no old numerical results were relabeled or credited.

## Scientific configuration

- Atlas samples 4 and 6 supply both host-clay Pc and fault-clay master Pc,
  converted at 40 degrees. Profile/table hashes are unchanged from v1.
- Host Pc endpoint remains Sg=0.57 with constant endpoint extension, not a
  hard saturation cap. Fault-clay master endpoint remains 0.80; sand is 0.95.
- Pc retains 100 pressure states and the original drainage/trapping physics.
- Kr now uses `adaptive10_endpoint_delta_v1`, pinned in CONFIG.json and
  METHOD.json. It uses at most ten shared gas/brine states, including endpoints,
  refining the largest observed Kr change. It sees only queried Kr values.
- The selector and replay source are byte-identical to the tested Oct 27
  implementation. It is a practical heuristic, not a guarantee of greater
  accuracy for every phase and core. No oracle/dense-reference selector is used.
- Geometry, intrinsic properties, local Kr laws, local sand Pc, numerical
  tolerances, corrected direct solver and acceptance checks remain unchanged.

## Integration evidence

The isolated six-run integration uses three saved representative geometries
(low_143, intermediate_167, high_144), each with both new Pc samples. It verifies
40-degree preparation, adaptive decision traces, exact replay microstates,
unchanged Pc100 versus the fixed-selector pilot, shared-state Kr agreement,
native solver lifetime, and actual reservoir imports of six core tables plus
both host-clay tables. Results are qualification evidence, not new-case credit.

All six runs passed in job 27293942 (3m03s; 2 CPUs / 8 GiB; peak about 3.44 GiB).
The tested grids had 30k, 40k and 90k cells. The legacy native-import environment
logged CPU-feature/cache warnings also seen in the parent pilot; import checks
passed, but portability to every cluster node is not established by this test.

`evidence/adaptive_integration` contains the receipts, allocation/accounting and
compact result archive for job 27293942. `integration_source` is the frozen test
source; `physics` retains the parent fixed-selector physics/pilot snapshot for
provenance, not as the new production entrypoint. No resident manager is added.

## Launch gate

The full campaign is NOT launched or authorized by this package. The old
campaign remains paused. The intended execution design is 32 cores per worker,
17 packets per case and a continuously replenished eight-case window, with
Slurm scheduling and no array throttle. This pilot does not qualify the future
runner's fine-core input generation, 32-core packet lifecycle/recovery, largest
grids, or population-wide memory. These remain explicit pre-launch checks.

External library identities and original sampling qualification are inherited
from the unchanged parent audit. This version checks the frozen snapshot but
does not claim to have rehashed every external PREDICT library.

Read-only verification:

```powershell
python -B freeze_configuration.py --verify
```

Never run the historical launchers under physics/integration_source as a
campaign launcher. Any deployment needs separate qualification and explicit
user launch authorization.
