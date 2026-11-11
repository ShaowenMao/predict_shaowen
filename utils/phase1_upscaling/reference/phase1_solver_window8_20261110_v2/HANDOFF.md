# Corrected Rolling64 Scheduling-Only Handoff

Source LOCK f7cb39e7b0934f08249a728333c972e90553566f3ae45d9553928c76ecaf97bd.
Frozen numerical/publication source unchanged:
phase1_solver_targeted_20261110_v1,
846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502.
User authorized up to64 outstanding workers, with independent small reporting
batches, no accepted-result recalculation, scientific-check changes or held retries.

Predecessor phase1_solver_window8_20261110_v1 passed31 compute tests/read-only
qualification28635308, but its actual overlapping pilot found a root-argument
bug BEFORE all numerical work. Old jobs28635657_0..7/28635664_0..7 FAILED at
the private-output guard; publishers28635659/28635665 FAILED before writes.
No preparation/probe/full-flow output exists under either possible output root
for any of the16 keys. Failed controller-root claims and isolation evidence
are preserved. This is one failed operational preflight per core, not permission
to reset attempt counts or repeat saved science. This source corrects only the
launcher argument and one-time proven-unstarted pilot reconciliation.

34 local unit tests passed, adding actual sbatch-argument regression with a
different controller/root, rejection of a wrong root on compute, and retention
of original pilot submission records. Cross-root compute check28636221 is
submitted held/released with verified advanced CPU allocation/spooled launcher:
1CPU/2GiB/30min, mit_normal and mit_amf_advanced_cpu account and QoS.
It must write ARGUMENT_PASS.json proving the launcher receives the legacy
numerical output root, not this controller root. No numerical work in that test.

Next CLI: set PHASE1_RUN_ROOT to this source's output, use python3.12 -B:
window.py submit-qualify performs compute terminal/no-science reconciliation,
pins original failure evidence in MISROUTE_PASS.json and repeats read-only tests.
window.py pilot then submits exactly the16 unstarted pilot keys, only once,
with *_rootfix1 submission labels. Original batch003/004 LAUNCHED records are
NEVER replaced. Authoritative corrected maps are output/repairs/003 and004.
window.py submit-adopt verifies both pilots through canonical publication before
ADOPTED.json and filling the64-worker window. No adoption has occurred yet.

All future batches005..019 use original canonical admission/intent/ACK paths
and unchanged legacy worker/publisher launchers. At most64 running/pending/
held/unknown numerical tasks; additional short finite completion handoffs refill
without a global batch/publication barrier. Original submit.lock serializes all
admission; publication uses original locks. Known strict scientific holds stay
isolated, unclassified control/integrity errors STOP new admission. Slurm handles
unrelated resource use. No manager daemon or desktop automation is added.
At last accepted status2,517 full cases;142 unfinished cores across75 cases.

## Corrected Pilot Released

Cross-root check28636221 COMPLETED0:0 in1s: ARGUMENT_PASS.json proves actual
launcher root is the unchanged numerical output directory. Qualification28636410
COMPLETED0:0 in6s;34 compute tests and terminal/no-science audit passed.
MISROUTE_PASS.json pins all32 original claim/isolation files and explicitly
records one prior operational failed launch per core, zero scientific work.

Corrected batch003 array28636725 / publisher28636726 and batch004 array28636731 /
publisher28636732 submitted held and allocation/spooled launcher checked before
release. They use exactly the old numerical code/2CPUs/18GiB request and old
publication code/1CPU/2GiB request. Old failed jobs and LAUNCHED records remain
untouched. Corrected maps in output/repairs/003/LAUNCHED.json and004/LAUNCHED.json
are authoritative for these two batches. No automatic expansion adopted yet.

## Pilot Lifecycle Passed; Adoption Submitted

Both corrected pilot arrays finished. Batch003 published7 and isolated1;
batch004 published4 and isolated4. All five holds are evidenced strict-reference
failures and remain untouched. Publishers28636726 and28636732 COMPLETED0:0 in
4m23s and3m21s. Both canonical batch PASS records passed; all11 published receipt
identities match their private qualification. Completed flow calls repeated=0.
Latest batch003 closeout reports2,528 complete cases. The remaining131 cores
include12 known isolated cores in this targeted scope and119 not yet attempted.

Adoption job28638567 submitted through tested window.py submit-adopt, held and
verified before release: mit_normal, mit_amf_advanced_cpu account and QoS,
1CPU/2GiB/30min, exact frozen v2 launcher. It was PENDING at17:43 cluster time
on2026-11-10. Do not submit again. Adoption is established only by ADOPTED.json
and its compute lifecycle, not by this submission note. No source or numerical
checks changed; old campaign states stay terminal and difficult cores stay held.

## Rolling Window Adopted

Adoption28638567 COMPLETED0:0 in55s. ADOPTED.json pins this v2 source and the
unchanged numerical source; events/28638567.json records batches005..012
admitted from zero outstanding workers, totaling64 numerical tasks.

Batch: worker / publisher / finite refill handoff
- 005:28638804 /28638805 /28638806
- 006:28638807 /28638808 /28638809
- 007:28638812 /28638813 /28638814
- 008:28638815 /28638816 /28638817
- 009:28638821 /28638824 /28638825
- 010:28638829 /28638830 /28638831
- 011:28638833 /28638834 /28638835
- 012:28638837 /28638838 /28638839

All actual numerical and publisher allocations verified from saved allocation
records: advanced CPU account/QoS, mit_normal, numerical2CPUs/18GiB and
publisher1CPU/2GiB, no array throttle, released once. Numerical execution has
started. Batch006 subsequently finished2 privately qualified and6 isolated;
its publisher/refill jobs became scheduler-Priority pending. Batch005 still
had one running probe, so batch006 can independently trigger the next refill.
No STOP or controller error observed. Automatic post-adoption refill has not
yet executed in this snapshot; do not create another controller or submit seed.
Canonical accepted total remains2,528 until later publication receipts exist.
