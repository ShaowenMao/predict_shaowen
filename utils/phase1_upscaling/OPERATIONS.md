# Operation And Recovery

## First Pass: Simple Finite Slurm Handoffs

1. Freeze the case catalog, exact stochastic assignments, physical profiles,
   scientific method and source/runtime hashes once. Never redraw during retry.
2. At admission, read only the case's scenario manifest and seal its 522-record
   plan. Order by slot: window 1 strikes 1..87, then windows 2..6.
3. Partition into 16 packets of 32 cores and one packet of 10. A worker processes
   its physical cores sequentially; "32 cores" does not mean 32 requested CPUs.
4. Keep a rolling eight-case window. Each terminal case gate refills a vacancy
   immediately; it does not wait for all eight cases. A partial case is terminal
   for first-pass scheduling but is not scientifically complete.
5. Submit finite worker arrays and `afterany` case gates. Failed tasks therefore
   cannot strand later cases behind an `afterok` dependency. There is no resident
   manager, desktop-monitor requirement, or second seed.
6. Close a full case only when all 522 exact receipts pass their identity checks.
   Future planning skips that canonical case marker without rescanning payloads.

First-pass v5 requests **2 CPUs, 18 GiB and two hours per packet**, no artificial
array throttle, and no automatic requeue. Eight full cases expose up to **136
queued/running packet tasks** (8 x 17), not 128. The explicit **128 outstanding
packet-reservation limit belongs to the later recovery controller**, not v5's
first-pass planner. Slurm decides which resource requests can run concurrently.

Always request `--partition=mit_normal --account=mit_amf_advanced_cpu
--qos=mit_amf_advanced_cpu`. The historical shared enforced association was
512 CPUs/2 TiB; confirm current limits before new deployment. Do not recreate a
second scheduler that fails when an unrelated movie/reservoir job appears.
Resource/priority/dependency waits are not software failures. A measured memory
warning is not a failed scientific result; real OOMs are isolated for recovery.

## Submission Safety

The original `slurm.py` stores an immutable intent before `sbatch --hold`, then
an ACK, actual allocation and release receipt. Verify account, QoS, partition,
CPU count, memory, wall time, array extent/no throttle, dependency and spooled
launcher before release. Arm the successor before releasing its worker array.

An ACK is reused, never resubmitted. A lost ACK must reconcile to exactly one
Slurm job using the saved identity/comment and accounting. Zero or multiple
matches are uncertainty, not permission to submit. Never delete intents to
make a retry possible. The final targeted accounting fix uses raw Slurm job IDs
where necessary; carry that regression into future adapters.

## Worker Stages And Durable Boundaries

| Stage | Work and saved evidence |
| --- | --- |
| Input | Exact PREDICT replay, original draw/geometry checks, record and INPUT_PASS |
| Pc/drainage | Original 100 pressure states and trapping; verified state cache avoids repeating drainage for adaptive Kr queries |
| Kr | Adaptive shared-state selection; intrinsic and phase-specific solves with existing numerical/conservation checks |
| Flow checkpoint | Saved call history, curve files, selection, summary and CORE_PASS |
| Export | Validated reservoir table `Sg,Krg,Krw,PcPa`; original input/output identity retained |
| Native import | Actual JutulDarcy table import and interpolation/derivative checks |
| Publication | Member-checksummed compact ZIP, immutable canonical receipt, then full-case closeout |

Slurm `COMPLETED` alone is not acceptance. Each physical core must pass the full
chain. A probe that passes one failed pressure system is diagnostic only until
complete recovery, export, native import, archive and publication also pass.

## Recovery Decision Table

| Existing evidence | Correct action |
| --- | --- |
| Valid accepted ZIP and receipt | Skip numerical work; never overwrite |
| Accepted ZIP, receipt missing | Verify members/identity and reconstruct the receipt; zero new solves |
| Saved CORE_PASS, no archive | Finish export/import/archive only; zero repeated flow calculations |
| Saved successful calls and interrupted later call | Prove exact input, method, terminality and prefix before resuming only missing calls |
| SOLVE_STARTED but unclear state/output | Hold and classify; missing DONE never proves safe recomputation |
| Proven unstarted core | Prepare/compute once from its original fixed assignment |
| Scientific/identity uncertainty | Isolate affected core, preserve files and attempt history; continue independent valid work |

The first-pass v5 code is intentionally conservative about interrupted cores.
Use the qualified recovery path, not a blind rerun of the original packet.
Completed calls are reusable only after the complete exact-input/operator and
saved-prefix checks. A force-killed core is not automatically resumable at an
arbitrary iteration; availability of a verified checkpoint determines the route.

Round two first verified saved flows, then recovered interrupted/unstarted work.
Its wider recovery-only case window avoids letting a few slow tails idle CPUs;
the 128 ceiling counts all original packet reservations until that case is
terminal. Round three used one-core workers and four-hour requests for the
remaining slow cores. Do not impose the eight-case first-pass window on sparse
late recovery or increase its worker ceiling casually.

The tested time-budget guard uses the actual allocation end, reserves 600 seconds
for finalization and 30 seconds for exit, and confirms owned child processes are
terminal before export. Round-two/three stage caps are 6900/13800 seconds. Never
retry an enclosing mutating stage under a read-timeout handler.

## Exceptional Recovery

Read-only classification precedes targeted batches. The retained paths cover:

- Positive saturation gaps below 1e-12: preserve distinct nodes; no merging,
  smoothing or saturation changes; all other export/native checks remain.
- Iterative/reference failures: preserve and reuse passing calls; independently
  verify a bounded solver change on the actual saved failed system first.
- Precision plateau: precise operator/audit and independent reference checks,
  not relaxed residual/conservation requirements.
- Final direct references: reuse the saved passing primary, compute one new
  independent reference, then only genuinely missing later flow calls.

These are qualified **scope-specific** paths, not automatic blanket retries.
Keep unsuccessful-attempt counts across versions and bound attempts. The normal
policy was at most three unsuccessful recoveries of a condition; the final four
received separate explicit authorization for one additional reference-only test.
Never reset that history or infer further authorization. Escalate unresolved
scientific or submission uncertainty. Preserve original failure records.

## Results And Storage

Canonical root on Engaging:

```text
/orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_campaign_20261030_v1
  results/<case_id>/core001.zip ... core522.zip
  runs/<case_id>/done/coreNNN.json
  complete_cases/<case_id>.json
  plans/<case_id>.json
  plans/<case_id>.seal.json
```

The compact ZIP contains input proofs, `flow/pc100.csv`, `flow/kr.csv`, saved
selection/audit information, `export/sgof.csv`, export/native passes and a member
checksum manifest. Export Pc is in **Pa**, not bar. Read those saved tables for
reservoir assembly or plotting; do not rerun upscaling to obtain them.

Only a worker's private temporary core directory may be retired after full
archive verification. Keep failed/partial outputs, original inputs, all receipts
and source pins. Never apply blanket cleanup to the campaign. Stage verified
software on node-local scratch; do heavy inventories/hashing on compute, not
login nodes. Completion planning should get cheaper as case markers accumulate.

Native-ready tables are not a promise that an arbitrary reservoir deck is ready
or will converge. Preserve the correct sample-specific host-clay tables, units,
phase mapping, window/strike order and pre-breakthrough Pc convention at import.
