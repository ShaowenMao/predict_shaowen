# Recovery-Only Wider Case Window, 2026-11-09

User authorized relaxing the eight-case limit for recovery only, retaining the
128-worker ceiling and safeguards, without interrupting current workers or
recalculating completed results. No new manager or monitor was created.

P = /orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_campaign_20261030_v1
H = P/round2_20261109_v1; existing solve output remains H/solve.
C4 = /orcd/data/juanes/001/shaowen/phase1_round2_20261109_v4
C4 LOCK = 988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019
Parent C3 LOCK = facf4d6fb3f93f144c19dfd6862e7cf45c24803b6333b4d69d6031304adc2c6a
Method = ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832
Configuration = 7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36

## Change And Qualification

Only numerical-recovery admission expands beyond eight cases. The same 128
packet-reservation ceiling applies: every original packet remains charged until
its entire case is terminal. Completed packets within a live case do NOT free
reservations early. Cases remain ordered and a whole next case must fit. The
finish/pilot case limits, first-pass code, worker/science/Slurm helpers, 2 CPU /
18 GiB / 120min allocation, exact-input guards, time-budget fix, saved-output
handling, claims and 440 submitted-task guard are unchanged.

Original solve SCOPE/plans/seals still have their C3 source and max_cases=8;
do not rewrite them. Append-only H/ADOPTION_recovery_window_20261109_v1.json
authorizes the effective wider recovery window and C4 future submissions.
Parent workers keep their original source; existing worker ACKs are validated
and never resubmitted. Replacement-gate mappings preserve original LAUNCHED files.

Local suite: 104 tests, 102 passed and two Linux-only skips. Preliminary local
test setup needed a USER fixture and copying the unchanged LOCK-pinned pilot
plan; these were resolved before compute qualification. The sealed source was
not edited. Build recreation must also copy the parent LOCK['plans'] fixtures.
Compute qualification 28474289_0 passed all 104 tests and the copied saved-flow
export/native/archive lifecycle, 0:0 in 25s, MaxRSS 608364K. No numerical solves;
original archives and an already accepted copy stayed unchanged. No unsuccessful
compute qualification or production recovery attempt occurred in this transition.

## Adoption

Only eight pending old control gates were temporarily held. Operator 28475095
completed 0:0 in 57s, MaxRSS 48524K. It verified held replacement allocations and
launchers, retired only those old pending gates, and preserved 681 protected
authority/claim/archive files. All eight existing numerical arrays were untouched.
Proofs: H/transitions/recovery_window_20261109_v1/TRANSITION_PASS.json and
RELEASES_PASS.json. No completed calculation was repeated.

Original worker arrays: 28465767,28466680,28469889,28471571,28471573,28471583,
28471595,28472298. Corresponding C4 replacement gates: 28475313,28475315,
28475318,28475321,28475322,28475323,28475324,28475325.
Finite refill job 28475326 is released with afterany:28475095, 1 CPU/1 GiB/45min,
mit_normal and mit_amf_advanced_cpu account/QoS. Do not submit it again.

At 17:13 UTC the refill job was scheduler-pending. Existing solve STATE was
cursor 15, eight active cases, 78 conservative packet reservations and 35 running
workers, with no control errors. Refresh live state; adoption is verified but
the expanded live window had not yet been observed at this snapshot.

All held exceptions remain untouched. First pass remains terminal; older
campaigns remain paused. Accepted archives and immutable numerical claims
continue to exclude completed or already assigned work.

## Live Expansion Verified

Refill 28475326 completed 0:0 in 13s at 17:14:01 UTC. By 17:16 UTC, solve STATE
advanced to cursor 20 with 13 active cases and 126 conservative worker-packet
reservations, below the unchanged 128 ceiling. It admitted five further cases,
s01_c021_pc06_ca40_IND_001 through _005, with 48 new workers across arrays
28476232 (10), 28476235 (11), 28476238 (11), 28476240 (9), and 28476242 (7).
Their gates are 28476234,28476236,28476239,28476241,28476243.

All five immutable held-allocation records and release receipts were checked:
mit_normal, mit_amf_advanced_cpu account/QoS, 2 CPUs, 18 GiB, 120min, C4 launcher,
and no array throttle. Live new arrays were scheduler-pending, not user-held or
dependency-blocked. Original eight arrays accounted for 45 completed and 33
running workers at the final accounting check, with no cancelled/failed tasks.
No case control ERROR records were present. The wider admission and refill
lifecycle are verified; new-worker completion and sustained speedup are not yet
measured. No further bootstrap or operator submission is needed.

## Round Three Successor, 2026-11-10

Round two reached ROUND_TWO_TERMINAL.json at 09:46 UTC with 6083/6478 solve
cores accepted. User authorized round three for the remaining 13 saved flows,
158 interrupted and 224 unstarted cores, leaving older held exceptions alone.
Follow ../phase1_round3_20261110_v1/HANDOFF.md and the campaign's new
round3_20261110_v1 output. New source LOCK is
2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b.
Qualification passed all 91 compute tests. All 13 saved outputs are now
accepted without numerical recalculation, original files preserved. Solve
bootstrap 28592054 admitted the first 121 workers under the unchanged 128
ceiling. Round three uses one core per worker, 2 CPUs/18 GiB and four hours,
with a matching bounded stage budget. Do not restart round two or its controls.
