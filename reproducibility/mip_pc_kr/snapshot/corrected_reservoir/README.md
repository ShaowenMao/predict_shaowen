# Corrected Fault-Hysteresis 15-Case Study

Status at 2026-09-11 16:09 UTC: all 15 full reservoir jobs are RUNNING. All 15
preflights and the shared 3,019-assertion regression suite passed. The simulations
have only just started; this is not yet a conclusion about convergence or leakage.

## Scope

Isolated comparison requested by the user. Original production files, properties,
running jobs, and completed results are not modified.

- Three original-property controls: low, intermediate, and high leakage-risk cases.
- Six MIP Pc + MIP Kr cases: each risk at WRR30 and WRR60.
- Six MIP Pc + production-style dynamic Kr hybrid cases: each risk at WRR30 and WRR60.
- All twelve MIP/hybrid cases retain physical pre-breakthrough saturation. The three
  controls intentionally retain the original production Pc and Kr tables unchanged.
- Existing same-draw property files are checksum-pinned and reused, not regenerated.
- All 15 new reservoir simulations start from time zero, with no checkpoint reuse.

## Correction And Checks

The isolated runtime is based on production commit
`ed81af8e88fcd2a7720f26872eef1d16d63469f5`; the corrected commit is
`c403d9dd1bbb424ddfae9ae659d900c26c84e59d`. The dependency Manifest.toml is unchanged.

Only three runtime files differ: the relative-permeability implementation, the
explicit-fault importer, and production QoI gas-mobility accounting. Region ids
9:530 explicitly bypass hysteresis and use native drainage curves for both phases.
Regions 1:8 retain the original host hysteresis. The partitioned-model constructor
also preserves these settings. This is a runtime correction, not a Pc/Kr table edit.

Each new model must pass the existing assembly/table/geometry/initial-pressure/QoI
contract plus an added fault-mask, native constitutive-value, AD-derivative, and
QoI-branch test. Tests use all 530 region representatives and saturation histories
around the Killough activation threshold and endpoints. Expected host values come
from independent copies of the immutable original functions. Fault values must
match native drainage exactly. The mask must match all 32,190 PREDICT cells and
exclude all 2,132,892 host cells.

The shared regression job `22588250` passed all 3,019 assertions: 2,923 existing
relative-permeability checks, 89 inventory/QoI checks, and 7 new drainage-mask checks.
Its predecessor `22588018` stopped before
testing because the new clone lacked the git-ignored dependency manifest. The
original pinned manifest was copied unchanged before resubmitting.

## Submitted Jobs

All full jobs depend on both their own preflight and the shared regression job
with `afterok`. Independent preflights run alongside the regression suite to avoid
unnecessary serialization. There is no study array-concurrency throttle and
no wildcard `aftercorr`. Submission does not mean that validation or simulation
has completed. Use the Slurm job ids below for live status.

| Risk | Properties | Preflight | Full Simulation |
|---|---|---:|---:|
| low | production | 22588287_262 | 22588288_262 |
| low | mip WRR30 | 22588289_262 | 22588290_262 |
| low | mip WRR60 | 22588291_262 | 22588292_262 |
| low | hybrid WRR30 | 22588293_262 | 22588294_262 |
| low | hybrid WRR60 | 22588295_262 | 22588296_262 |
| intermediate | production | 22588297_1137 | 22588298_1137 |
| intermediate | mip WRR30 | 22588299_1137 | 22588300_1137 |
| intermediate | mip WRR60 | 22588301_1137 | 22588302_1137 |
| intermediate | hybrid WRR30 | 22588303_1137 | 22588304_1137 |
| intermediate | hybrid WRR60 | 22588306_1137 | 22588307_1137 |
| high | production | 22588309_848 | 22588310_848 |
| high | mip WRR30 | 22588311_848 | 22588312_848 |
| high | mip WRR60 | 22588313_848 | 22588314_848 |
| high | hybrid WRR30 | 22588315_848 | 22588316_848 |
| high | hybrid WRR60 | 22588317_848 | 22588318_848 |

Full jobs request 8 CPUs, 18 GB, and three days. Preflights request 8 CPUs, 24 GB,
and four hours. The full runs use the original production solver/timestep controls,
including per-iteration CPR updates, with host hysteresis retained. The initial
four-day scheduler request overlapped the September 15 maintenance reservation.
Only these pending study jobs were shortened to 72 hours to fit before maintenance;
no numerical controls were changed. Own-run checkpoints remain on project storage
if any case subsequently requires continuation.

All 15 assembled-model preflights passed, including the input-pair checks and native
fault-only constitutive/QoI tests. Full simulation completion remains separate from
these successful checks.

All 210 report states, schema-4 QoIs, rolling checkpoints, logs, and temporary files
are directed to project storage:

`/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/corrected_hysteresis_15case_20260911_v1`

No simulation output is intended for scratch. The project filesystem reported
approximately 37 TB available during setup.

## Reproducibility

`matrix.json` identifies all source property files and checksums; `submissions.json`
maps variants to jobs. Each remote variant contains its own campaign.toml,
locked.env, source scripts, INPUTS_SHA256.txt, and STUDY_SHA256.txt. Both original
input-pair audits and new fault-only hysteresis checks gate every full run.

`prepare.py prepare` creates the isolated launchers from existing verified study
sources. It refuses to rewrite a variant with a results directory. Submission is
separate and refuses a duplicate submission ledger. Never rerun submission blindly
after a partial failure; recover only the affected jobs and preserve the ledger.

The 15 pending preflights were briefly held before starting to finish the validator's
use of the initialized simulator state, then released. No production or legacy
comparison jobs were held or changed.
