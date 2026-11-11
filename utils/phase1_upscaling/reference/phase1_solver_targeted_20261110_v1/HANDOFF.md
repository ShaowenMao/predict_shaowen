# Targeted Solver Recovery

User authorized cores345/380/402 first, then small targeted batches of the
remaining held cores. Do not recalculate accepted or logged successful flows.
Old campaigns, other agents and reservoir jobs remain untouched.

Source LOCK: 846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502.
Parent: phase1_solver_finish3_20261110_v1. Its strict solver and full-flow
saved-prefix replay are byte-identical here. The probe enables its already
tested tighter reference option; physical operators and acceptance are unchanged.
30 local Python tests and 26 local Julia tests passed; compute repeats tests.

Fixed154 remaining keys from the original held173 audit, excluding the16
published export recoveries and the3 already-published solver recoveries.
Batch000 is core345 in s04_c005_pc04_ca40_IND_003 and cores380/402 in
s02_c024_pc06_ca40_IND_003. Subsequent batches contain up to8 keys, ordered
to finish cases. No automatic retry, source edits or new resident manager.
Each task privately probes only its failed gas call, then reuses original
successful calls plus that probe through full Pc/Kr/export/native/archive
verification. Failed tasks retain evidence and cannot be blindly resubmitted.
Publication is a separate locked operation, with exact archive/member checks,
original assignment reconciliation and completion sealing only at522 accepted.

Remote root: /orcd/data/juanes/001/shaowen/phase1_solver_targeted_20261110_v1
PHASE1_RUN_ROOT must be that root/output; use /usr/bin/python3.12 -B.
Tested CLI: submit.py <batch>. First-three PASS required before batch001.
Previous batch publication receipt required before subsequent admission.
Immutable submission intents/ACKs/allocations/releases live in output/submissions.
Never repeat an uncertain submission or a task with a claim/output directory.

Batch000 worker array28631396 (3 tasks, each2CPUs/18GiB/120min) and
publication28631397 (1CPU/2GiB/30min, afterany array) submitted held,
actual mit_normal/mit_amf_advanced_cpu account and QoS, memory, CPU,
no throttle/requeue, dependency and spooled launcher checked before release.
Before launch the campaign/recovery STATE active arrays were empty,154 cores
remained in86 cases and2,506 full cases were sealed. No new failures yet.
Shared filesystem19TiB available; only a bounded fixed scope is read.

## Batch000 Passed, Batch001 Submitted

All3 full pipelines passed; tasks28631396_0/1/2 completed0:0 in2m57/2m57/2m56,
MaxRSS1,837,760/1,869,936/1,845,292KiB. Original saved flow calls reused16/18/18,
plus one newly verified failed-gas probe per core; subsequent new calls4/2/2.
Thus11 missing flow evaluations completed,52 original successes never repeated.
All exports/native imports/archive checks passed. Publisher28631397 closed
both requested cases; output/batches/000/PASS.json records2,508 complete cases.
151 cores remain in84 partial cases. No scientific failures in this batch.

One batch001 submit preflight read PASS before NFS visibility caught up;
it exited before any new admission/intent/submission. After a fresh full PASS
read and proof that no targeted001 intent existed, submission was safely made.
This is not a numerical recovery failure and no mutating body was read-retried.

Batch001 array28631692 (8 tasks) and publisher28631693 admitted/released with
the same checked advanced allocation. Each of these8 cores is the final
missing position of its case. No further batches have been submitted yet.

At21:31 UTC batch001 remained scheduler-pending (Reason=None), publication
dependency-pending; no failure or user hold. Batch000 publisher completed0:0
in46s, MaxRSS127,532KiB. Canonical done receipts match all3 qualification
receipts and both case seals exist. Downloaded all3 compact archives/work
receipts plus batch/submission/publication evidence; every local archive
member and native/export/flow receipt reverified. No source changes after seal.

## 2026-11-10 21:47 UTC: Batch001 Reconciled, Batch002 Submitted

Batch001 finished:3/8 fully verified and published,5 isolated. Accepted keys:
s01_c001_pc04_ca40_IND_005_core117,
s01_c002_pc04_ca40_IND_003_core211,
s01_c002_pc04_ca40_IND_005_core254. Each closes its case. Canonical receipts
match the qualification receipts and all3 completion seals exist. Total2,511
full cases;148 missing cores across81 partial cases. The3 reused48 original
successful flow calls and their3 new strict probes, followed by12 missing
calls. No previously completed calls repeated. Successful workers took
3m53/3m54/3m53 with peak RSS4,139,108/2,941,644/2,674,840KiB.
Publication28631693 completed0:0 in3m37, MaxRSS193,728KiB.

Four isolated cores failed the unchanged reference mass-conservation target:
s01_c002_pc04_ca40_IND_007_core297,
s01_c005_pc04_ca40_IND_005_core090,
s01_c005_pc04_ca40_IND_006_core496,
s01_c022_pc04_ca40_IND_002_core033. Core s02_c002_pc04_ca40_IND_002_core129
passed its first failed-state probe, then failed the unchanged primary check
at a later gas state59. Its partial flow log and gas59_strict.toml are retained.
Each has one original unsuccessful core attempt plus one new unsuccessful
recovery attempt. Do not retry them or discard their saved partial results.
All5 have immutable isolation records; no identity or submission defect found.

User requested continuing the next small batch while leaving difficult cores
isolated. Refreshed source pin, original terminal/intent/ACK/release evidence,
empty old campaign active lists and absence of accepted outputs, claims or
prior attempts for every batch002 target. No prior batch002 submission existed.
Tested CLI submit.py2 launched array28633792 and publisher28633793. Actual
held allocations, spooled source launcher, no throttle/requeue and dependencies
checked before release: numerical2CPUs/18GiB/120min; publication1CPU/2GiB/30min;
mit_normal with mit_amf_advanced_cpu account and QoS. At21:47 UTC all8 tasks
were scheduler-pending (None), publisher dependency-pending. No batch003 launch.
Other jobs and paused campaigns unchanged. Source remains frozen and unchanged.

## Scheduling Successor Under Qualification

User approved a64-worker rolling window. Follow
../phase1_solver_window8_20261110_v2/HANDOFF.md for admission authority.
Numerical and publication code here is unchanged. Do not manually submit
further batches with the old serial CLI while this successor is active.
Batch002 accepted6 and isolated2, bringing full cases to2,517 (142 cores left).
The first scheduling-only pilot stopped before science due to a launcher-root
argument defect; its16 claims/errors live in window8_v1/output and are preserved.
Batch003/004 original LAUNCHED records refer to those failed preflights and
must not be replaced. Corrected, proven-unstarted pilot maps live under
window8_v2/output/repairs/003 and004: workers28636725/28636731, publishers
28636726/28636732. All34 tests, actual cross-root compute test and no-science
audit passed before this single correction. Automatic64-worker expansion is
still gated on full pilot publication/adoption; see successor for live status.

## Rolling Successor Adopted

The corrected overlapping pilot published11 cores (7+4), with5 strict-reference
holds preserved, bringing full cases to2,528. Scheduling-only v2 adoption
28638567 COMPLETED0:0; ADOPTED.json and the first event prove batches005..012
were submitted as64 independent tasks. Follow window8_v2/HANDOFF.md and its
events/admissions/STOP evidence. The finite Slurm handoffs now own continuation;
do not use this old serial submission CLI or repeat any pilot/accepted core.
Frozen numerical and publication source here remains unchanged.
