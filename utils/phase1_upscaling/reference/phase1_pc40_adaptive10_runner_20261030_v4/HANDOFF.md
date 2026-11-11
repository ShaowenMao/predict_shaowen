# Verified drainage cache: 32-core packet integration

2026-10-31 01:35 UTC verification. Candidate packet qualification PASSED.
All 28 local and compute unit tests and the real 32-core lifecycle passed.
Production remains on v3; no source switch, cancellations or production writes.

Source LOCK: 709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8.
Predecessor: phase1_pc40_adaptive10_runner_20261030_v3,
4497359d39e01a47ce830ca4495a9682ba330b210eaa0856b785c45761d551c5.
Cache source: phase1_adaptive_statecache_20261030_v1,
c1d92f53286c7444c3c0df09080654936405ee4eb4596c94f94dcab4832b1d0b.
Frozen configuration and numerical method identities are unchanged and recorded
in LOCK.json. Cache implementation hashes are recorded separately in flow receipts.

## Narrow change

- flow_packet.jl calls the qualified cache implementation without the test-only
  reference replay, and checks cache evidence before issuing a flow receipt.
- worker.py adds validation of cache implementation/state receipts. Proven v3
  saved flow remains recoverable without another solve; old accepted archives
  remain accepted. Two actual old sample-4/sample-6 archives passed read-only
  compatibility checks locally, including the saved-flow contract.
- control.py, planning.py, slurm.py, common.py, preparation, export and native
  import code are byte-identical to v3. The rolling eight-case window, standard
  17 packets/case, compact storage and two-CPU/18-GiB allocation are unchanged.
- The launcher adds only a bounded integration-test entrypoint. It still reads
  code routing from RUN.json. Do not change production RUN.json blindly: queued
  legacy launchers also consult that file. Future-only transition is separate work.

## Fixed qualification work

One 32-core packet from s01_c003_pc06_ca40_IND_007, using only previously prepared
but numerically UNSTARTED positions. All 17 original tasks are terminal and the
case is no longer active. Accepted results, completed flow and uncertain/held
calculations are excluded. Exact slots and original input checksums are frozen in
PILOT_INPUTS.json; ORIGINAL_PLAN.json preserves the production assignment identity.
The qualification plan regroups only this fixed subset, not the production plan.
The original production files remain read-only and are rechecked at each stage.
Results stay under this candidate's numerical/ directory, not production.

| Stage | Job | Allocation | Gate |
| --- | --- | --- | --- |
| Unit tests | 27397682 | 1 CPU, 1 GiB, 10 min | COMPLETED 0:0, 2s; 28 tests passed |
| Save two completed flow results | 27397683_0 | 2 CPUs, 18 GiB, 2h | Successful unit receipt |
| Resume and finish 32 cores | 27397684_0 | 2 CPUs, 18 GiB, 2h | Successful checkpoint and unchanged saved results |
| Archive audit and accepted-packet no-op | 27397685_0 | 2 CPUs, 18 GiB, 10 min | Successful resumed packet |

All submissions were held, actual allocation/spooled launcher verified, and
released using mit_normal plus account/QoS mit_amf_advanced_cpu, no array throttle.
Dependencies trigger finite test stages only; each stage additionally requires a
successful predecessor. No daemon, manager, production seed or monitor was added.

## Qualification result and remaining transition

numerical/PACKET_QUALIFIED_PASS.json verifies all 32 compact archives, scientific
checks, export/native import, exact preservation of the first two saved flow
results, and a real accepted-packet no-op. Production was not modified or adopted.
Jobs 27397683_0, 27397684_0 and 27397685_0 completed 0:0 in 7m03s, 21m47s and
8s respectively. Combined numerical-worker elapsed was 28m50s, including the
deliberate checkpoint and repeated input preparation, not repeated flow solves.
Whole-job measured peak RSS was 4560080 KiB (4.35 GiB), within 18 GiB requested.
Grids ranged from 20000 to 140000 cells. These 32 cores used the fast MIP path;
the preceding separate three-core qualification covered the slow path.
After-GC logs showed live Julia memory stabilizing near 22 MB and RSS near
1.17 GiB at the end of the resumed packet; no growing retained cache was seen
in this test. This is not a whole-campaign speed benchmark.

The remaining work is a reconciled, future-job-only production transition.
Do not blindly reroute RUN.json: already queued legacy launchers also read it.
Existing production jobs, accepted results and the paused old campaign remain
unchanged. No failed compute qualification or recovery attempts occurred.

## Qualification gates retained for audit

Require numerical/PACKET_QUALIFIED_PASS.json, all 32 compact archives and their
checksums, exact preservation of the first two saved flow results, real accepted
packet no-op, memory/elapsed accounting, and no unchanged-science violations.
Inspect per-core and after-GC memory logs before planning production adoption.
Do not call the candidate qualified from Slurm exit alone. Do not rerun enclosing
submission or numerical bodies on a timeout; reconcile saved checkpoints first.
An initial local build stopped on a whitespace-match assertion before sealing or
submission; corrected without any production effect. No compute recovery attempts
have been made. The old campaign remains paused.
