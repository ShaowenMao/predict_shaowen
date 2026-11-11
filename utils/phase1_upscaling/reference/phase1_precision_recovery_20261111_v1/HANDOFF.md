# Three-Core Precision Recovery Qualification

Scope: core253 of s04_c002_pc04_ca40_IND_001, core283 of
s01_c002_pc04_ca40_IND_008, and core422 of s04_c005_pc06_ca40_IND_001.
Only private complete recovery qualification is authorized. No production
publication/adoption, no broader recovery, no core229 retry or new manager.

Source LOCK SHA256:
`1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82`.
Saved diagnostic source:
`aea89e68590c31f60d060a40c009b82ee76df0a2983d7fa97a9da0af4e44463a`.
Scientific method/configuration and original successful flow prefixes remain
unchanged. The three saved primary/reference pressures and failed-state fluxes
are reused, not recalculated. Exact inputs, state/operator hashes, saved receipt
and pressure checksums are required. Re-audit uses the same thresholds with
256-bit face arithmetic and an independently assembled 384-bit operator.
The old binary128 audit disagreement remains explicit in the evidence, not
silently ignored. New, previously uncomputed gas states use the isolated accurate
correction path and both strict checks. Brine, Pc100, adaptive10, export and native
import are unchanged. All original evidence is protected before and after.

Local checks: 12 Python guards, 37 Julia arithmetic/gate/prefix checks passed;
all Julia sources parsed. A compute synthetic integration qualification must
pass before the three end-to-end jobs. Qualified outputs are private
`output/<key>/verified_core.zip` and `QUALIFIED_PASS.json`, not canonical results.
Submission uses the tested immutable-intent/ACK held-release helper: explicit
mit_normal and mit_amf_advanced_cpu account/QoS, actual launcher and allocation
checks, no array throttle, no requeue. Qualification: 2 CPUs/4 GiB/20 minutes;
full recoveries: 2 CPUs/18 GiB/2 hours each. No automatic retries.

The preceding original and targeted scientific failures (two per core), the
successful precision diagnostic, and older launcher-only failures remain
preserved. This is one first complete-recovery qualification per core, not a
reset of unsuccessful recovery counts. Unrelated jobs are untouched.

## Compute Qualification Passed

Job28653743 completed0:0 in1m00s, MaxRSS1,159,500KiB. All67 checks passed:
12Python guards,37Julia arithmetic/gate/prefix checks,18actual frozen-solver
integration checks including re-audit/reuse and mismatch rejection. The
qualification receipt pins the source above. No production core solved.

## Complete Recovery Qualified: Three Of Three

Array28654025_0..2 completed0:0. All three have explicit QUALIFIED_PASS.json,
native-import PASS, complete Pc100/adaptive10 curves and independently read,
member-checksummed private archives. CLOSEOUT.json records receipts and accounting.

| Core | Original calls reused | Saved precision call reused | New missing calls | Job elapsed | Peak RSS |
| --- | ---: | ---: | ---: | --- | ---: |
| 253 | 20 | 1 | 0 | 1m37s | 1,219,808 KiB |
| 283 | 16 | 1 | 4 | 2m09s | 3,473,700 KiB |
| 422 | 20 | 1 | 0 | 1m36s | 1,236,356 KiB |

Total59 saved passing flow calls reused,4 missing calls computed,0 completed
flow calls repeated. Each saved precision pressure passed fresh256-bit and
independent384-bit checks against the exact reconstructed operator, without a
new solve. Each job protected18 original/diagnostic files, all unchanged.
The four new calls were the final two brine/gas pairs for core283. Its gas calls
also used the newly qualified accurate path with distinct primary/reference
checks. The old binary128 audit disagreement remains recorded explicitly.
Scientific thresholds, input draws, physical conductances,40-degree profiles,
Pc100/adaptive10, export and native acceptance rules were not relaxed.

Private outputs: output/<key>/verified_core.zip. No canonical result or completion
receipt was written; production remains2,546 complete cases until separately
authorized publication. No core229 retry, broader recovery, source adoption,
resident manager, or change to unrelated/production jobs. Next, if authorized:
publish these verified archives without solving again, then consider a small
targeted expansion. Do not rerun these three qualification jobs.
