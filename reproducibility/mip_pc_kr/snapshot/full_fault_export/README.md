# One-Fault Pc100/Kr10 and Dispatcher Qualification

Scope: `s03_c022_case12`, 522 original placements, WRR60, physical pre-breakthrough
saturation, outlet-connected brine trapping, dip-parallel corrected direct flow.
Only Kr sampling changes. Old production, old results and paused controllers are
read-only. This is not a 2430-case launch or reservoir time-marching run.

## Gates

1. Verify the accepted 522-core reference receipt chain, fine inputs, geometry,
   source identity, original draw/seed mapping and WRR60 paired assumptions.
2. Canonicalize only core-label provenance and its geometry hash link for exact
   reuse. Byte-compare canonical inputs on a hash match. Keep differing archived
   floating-point representations separate rather than changing any Pc values.
3. Derive Kr at ten selected existing states without rerunning MIP or flow.
   Retain all Pc values. Record sampling errors and flags, never hide them or
   treat a large approximation error as a failed numerical solve.
4. Test partial-batch recovery: save two owners, then resume the same batch and
   verify those exact accepted artifacts are reused unchanged.
5. Exercise the source-pinned warm Julia worker through the new dispatcher with
   one four-record canary, using its qualified cache and unchanged algorithms.
6. Assemble all 522 placements, round-trip MATLAB files, preserve all non-fault
   inputs and absolute permeability/porosity. Check all native tables and AD
   derivatives, then assemble the full grid with host hysteresis retained and
   fault hysteresis disabled. No physical time marching.

## Dispatch and Resources

Finite, source-locked DAG; maximum eight outstanding qualification jobs. Each
job explicitly uses `mit_normal`, account and QoS `mit_amf_advanced_cpu` and its
actual allocation is checked after submission and inside the job. No array
throttle is added. Submission intents precede sbatch; returned IDs are saved
before verification. An uncertain submission is held unless a unique scheduler
token establishes its identity. A filesystem lock prevents concurrent dispatch.

The qualification admission check counts running/configuring/completing jobs,
all eligible external pending jobs, and all its own pending reservations.
External dependency-blocked and user-held jobs are explicitly recorded, not
treated as current reservations: counting every future stage of another DAG
would permanently block a small qualification even while CPUs are idle. It
rechecks before every submission. This is a 500-CPU/1984-GiB snapshot ceiling,
not a guarantee against independent future submissions; Slurm's advanced
association remains the hard account-wide guard. The frozen parent campaign
contract is not silently edited by this qualification policy.

Stage allocations: audit 1024 MiB, derive 512 MiB, warm canary 2816 MiB, MATLAB assembly
4096 MiB, native import 3072 MiB, full grid 12416 MiB. The last is rounded from
1.20 times the prior 10.036-GiB whole-job peak. Whole-job cgroup peak and limit
events are measured for every stage; acceptance requires zero limit events and
at least 1.05 times peak within the reservation. No blanket per-core increase.

Unqualified resource classes, full-campaign owner scheduling and reservoir QoI
accuracy remain distinct gates. This qualification does not claim those pass.

## v2 Audit-Only Repair

The v1 scientific audit completed all 522 inputs successfully. Its 512-MiB
whole-job memory gate did not pass; that report and all source calculations are
preserved. v2 distinguishes the archived scientific flow-log digest from full
file hashes and verifies both. Streaming hash readers advise the kernel to drop
already-consumed file-cache pages. Tests require identical hashes, headers and
byte counts to the old reader. Only the audit allocation is raised to 1024 MiB;
no numerical worker or per-core allocation changes. This reruns verification,
not MIP or flow calculations, and records memory.stat and process status.
