# Fixed Precision-Plateau Recovery Batches

User authorized publication of the three qualified cores, followed by targeted
recovery of the other cores with the same precision issue. Core229 must stay
isolated and no completed phase calculations may be repeated.

LOCK SHA256: `ada40e545387809fdedd8afd1781f9c75f2f10727178611baa0fcaabf792ab80`.
Parent complete-recovery pin:
`1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82`.
SCOPE.json fixes38 stable-reference-plateau cores:3 publication-only pilot cores
and35 further recoveries, in disjoint8/8/8/8/3 batches. The remaining18 held cores,
including core229 and the separate later-state failure, are excluded.

The solver, precise audit, strict thresholds, complete-flow recovery Julia code,
and failed-state precision experiment are byte-identical to qualified parents.
Only orchestration and bounded publication scope are adapted. Each new core first
checks its saved exact input/operator and original successful prefix, makes one
bounded precision attempt at its previously failed state, then reuses that result
and all original successful calls during complete recovery. It computes only
missing later states. All original evidence and partial new outputs are retained.
The two prior scientific failures per core are preserved; this is the third
bounded attempt, not a retry-count reset. No automatic retries or resident manager.

Local28Python guards/publication tests passed, including all three actual pilot
archives, conflict rejection, idempotent promotion and disjoint scope. Julia
syntax/arithmetic/strict-gate/saved-prefix checks passed. Compute qualification
must pass before any publication or new recovery. Actual synthetic integration,
native-ready archive checks and all28Python tests run on compute.

CLI: driver.py submit-qualify; then submit-pilot. After all three publications
pass, submit-batch0. After its full lifecycle/publication passes, submit-batch1..4
may run concurrently. Each batch is one finite array and one afterany publisher;
failed cores stay held while independently qualified results can publish.
Publisher performs no numerical calculations. It verifies every archive member,
exact assignments and sibling receipts before immutable promotion/case closeout.
Existing canonical locks, immutable intent/ACK/allocation records and source pins
are retained. Uncertain submission or publication is never blindly repeated.

All new jobs explicitly use mit_normal and mit_amf_advanced_cpu account/QoS,
held allocation/launcher verification before release, no throttle or requeue.
Qualification2CPUs/4GiB/20min; recoveries2CPUs/18GiB/2h; publishers1CPU/2GiB/2h.
Other campaigns and reservoir jobs are untouched. No full campaign rescan on
login nodes; full sibling/archive audits run only on compute for affected cases.

## Pilot Publication Completed

Compute qualification28655933 passed83 checks (28Python,37Julia unit/gates,
18actual solver/reuse integration), plus the three real qualified archives;
COMPLETED0:0 in1m00s, MaxRSS1,188,736KiB.
Publication28656479 passed for all three saved archives. It performed zero new
solves and canonically completed their three cases. Total now2,549/2,592 full
cases. output/batches/pilot.json and per-key published/PASS.json are authority.
The35 remaining precision-plateau targets are still distinct from18 unrelated
held cores. First8-core batch0 is now authorized by the completed pilot gate.

## First Eight-Core Batch Passed And Published

Recovery array28657015_0..7 passed complete recovery, export/native import and
archive validation for all8. Runtime1m35s..2m51s, MaxRSS1,209,748..3,390,296KiB.
It reused128 original passing phase calls, computed40 missing calls (including
8 formerly failed states), and repeated0 completed calls. Publisher28657016
completed all8 case closeouts; total2,557 complete cases. The remaining27 cores
can now run as independent batches1..4, without waiting between their numerical
arrays. Every batch retains its own afterany verified-publication job.

## Remaining Batches Released

Batch1: recovery28658620, publisher28658622,8cores.
Batch2: recovery28658640, publisher28658641,8cores.
Batch3: recovery28658654, publisher28658655,8cores.
Batch4: recovery28658851, publisher28658852,3cores.
All use the fixed source/scope above, held allocation and launcher verification,
then release through the tested CLI. No successful core is eligible for numerical
resubmission. Actual jobs and allocations are recorded in submissions ACKs.

## All35 Additional Recoveries Qualified

All35 have COMPLETE.json and QUALIFIED_PASS.json; no new scientific failure.
They reused560 original passing phase calls, computed175 previously missing
phase calls, and repeated0 completed phase calls. Per-job recorded elapsed
82.13..193.07s (Slurm startup/teardown is additional). Publication jobs continue
independently; count canonical published/PASS.json, not Slurm success alone.

## Final Closeout: All38 Published

All3 pilot archives and all35 additional recovered cores are canonically
published and verified. Every numerical array, qualification and publication
job completed0:0. All six batch receipts report all_passed=true, held=[].
This completed31 additional full cases:2,577/2,592 cases now complete.
Exactly18 classified held cores remain in15 partial cases, including core229;
none was retried or changed by this scope. No completed phase call was repeated.

output/FINAL_CLOSEOUT.json records all38 qualifications/publications, canonical
receipts,31 newly completed cases,18 excluded exceptions and Slurm accounting.
output/EVIDENCE.zip is the bounded metadata evidence mirror, SHA256
`d96d4dc7520b4f353d3576677b3db66b249f718f82d80858f68eebe5411608e8`.
Archives stay in the original canonical campaign results/<case>/coreNNN.zip;
their saved partials, diagnostic pressures and original failure records remain
preserved. No cleanup, tolerances, inputs, production runner or unrelated jobs
were changed. This finite recovery scope is finished; do not resubmit its jobs.
