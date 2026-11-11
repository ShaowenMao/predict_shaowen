# Round Three Recovery, 2026-11-10

User authorized: verify the 13 saved outputs without numerical recalculation,
then recover only round two's interrupted/unstarted cores. Prior 173 held
exceptions are outside this scope. No old campaign or completed case is reopened.

P = /orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_campaign_20261030_v1
New output = P/round3_20261110_v1; prior output = P/round2_20261109_v1.
New code = /orcd/data/juanes/001/shaowen/phase1_round3_20261110_v1
Source LOCK = 2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b
Parent LOCK = 988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019
Method = ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832
Configuration = 7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36

The last bounded round-two reconciliation found 6083/6478 numerical cores
accepted, 13 saved flows, 158 interrupted and 224 unstarted. Round two ended
2026-11-10 09:46 UTC, with no active cases or control errors. Two tasks had
NODE_FAIL; other tasks terminated with 0:0. Full-case markers counted 2434/2592.

The numerical recovery remainder uses one core per worker, four-hour jobs,
2 CPUs and 18 GiB, no array throttle, and the same conservative 128-worker
reservation ceiling. The stage cap is 13800s instead of 6900s, still bounded
by the actual allocation end minus the original 600s finalization reserve and
30s exit margin. No numerical method, tolerance, worker, Slurm helper, exact
input, native-import, archive acceptance or private-retirement logic changed.
The cache proof admits only the additional pinned round-two predecessors with
identical science. Old sources, outputs and numerical claims are preserved.

Finite workflow: read-only compute inventory; qualification; finish-only
workers; all saved archives and original files checked; one solve bootstrap;
case gates refill within the 128-worker budget. No new resident manager.
Uncertain submissions retain immutable intents and are never blindly repeated.

Local: 91 tests, 89 passed and two Linux-only skips. Initial local setup/test
adjustments were resolved before sealing; no compute failure at this snapshot.
Compute qualification 28591521_0 and inventory 28591523 submitted held, actual
allocations/launcher checked, and released. Both were Priority-pending at the
latest check. No production recovery submission or authorization record yet.
Do not submit duplicate qualification, inventory, or start jobs. Refresh their
canonical PASS/FAILED records and accounting before any further action.

## Qualification And Start

28591521_0 completed 0:0 in 49s; all 91 compute tests passed. The copied saved
flows passed export/native/archive and accepted no-op checks without numerical
solves. Qualification PASS SHA256:
2f419656c120344823e40b07807558e276b7ace21154e9c8682abe6867f062cb.
Read-only inventory 28591523 completed 0:0 in 1m53s. It skipped 70 completed
cases, verified 3344 archives in partial cases, and selected exactly 13 saved
flows, 158 interrupted and 224 unstarted cores. No ambiguous-output exclusions
were found in this remainder. Inventory PASS SHA256:
18a7d0785921314ad9b64513d592eb5ad46fe0bade4465445ea0d81d2be3c8c8.

New output AUTHORIZATION.json pins the source, qualification, inventory and
prior terminal proof. The root and both phase log directories were prepared
before launching the one finite start job 28591805, completed 0:0 in 7s.
Finish workers 28591852_0 and 28591854_0 cover 13 saved flows in
s03_c021_pc06_ca40_IND_001 and _008. Their afterany gates are 28591853 and
28591855. Initial live observation: workers scheduler-pending, gates dependency-
pending, no errors. Do not resubmit them or create another start job.

Only after all 13 original saved flows are archived and original checkpoint
files rehashed unchanged does SAVED_OUTPUTS_PASS.json allow the solve bootstrap.
The intended numerical scope is 382 genuinely interrupted/unstarted cores.

## Saved Outputs Accepted And Numerical Queue Armed

Finish workers completed 0:0 in 27s and 14s. Their summaries report 7/7 and 6/6
accepted, checkpoint_only=true, initial states all FINISH, and only runtime,
export and native stage logs. All 13 canonical production receipts are present.
Finish gates 28591853/28591855 completed 0:0 in 5s/6s. New output
SAVED_OUTPUTS_PASS.json confirms all 13 archive proofs, zero numerical solves,
and original_saved_files_unchanged=true. Full-case count remains 2434.

Exactly one solve bootstrap 28592054 was created by the handoff. It admitted
15 cases / 121 single-core packet reservations, below 128, from a sealed total
scope of 382 cores across 55 partial cases (158 interrupted, 224 unstarted).
Its first worker arrays are 28592128,28592130,28592132,28592134,28592137,
28592140,28592146,28592150,28592152,28592159,28592167,28592173,28592177,
28592180,28592182. All 15 actual held allocations and release receipts were
verified: mit_normal, advanced account/QoS, 2 CPUs, 18 GiB, 04:00:00, pinned
launcher, no array throttle. Initial snapshot: all 121 workers scheduler-
pending, all 15 case gates dependency-pending, no control errors. Automatic
case-gate refilling is armed; do not resubmit any existing job or bootstrap.
