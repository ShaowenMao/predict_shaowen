# Final Four Reference Recoveries V2

Continuation of the user's explicit authorization for one isolated direct-reference
test per core096,229,192,221, followed by full recovery/publication only on PASS.
Source8b61b7dabc42f57c858a2d137858fece866174ede388aa75fa6479137b2c0fbe.
Scientific code, inputs, saved-primary reuse and checks are byte-identical to
v1; only previous-job accounting is repaired to use JobIDRaw.46 local Python
tests passed, including the actual Slurm array/raw-ID records and missing/live/
duplicate rejection. Compute qualification28695227 now also runs the exact four
production preflights, without numerical solves, before synthetic lifecycle tests.

Preserved predecessor:source9e9c3a7ed5d1ebac3b94ce6af977baac55a0ee223e1d69135ddfd18eb1b66053,
array28694786_0..3 allFAILED1:0 in1s, publisher28694787 COMPLETED0:0 holding all.
Only ERROR.json exists in each predecessor core output; no primary/reference
calculation was performed. One operational preflight failure per core,3 prior
scientific failures per core, one fourth scientific attempt still authorized.
No automatic retry, no repeated completed calls, no new manager or old campaign.
All jobs explicitly mit_normal and advanced CPU account/QoS; held allocation,
source and launcher checks. Qualification2CPUs/4GiB/20min. Numerical jobs are not
yet submitted; require this version's compute PASS. Unrelated jobs untouched.

Qualification28695227 COMPLETED0:0 in1m09s,MaxRSS1444156KiB, including46 Python
tests,219 Julia checks, actual4 saved-input/accounting preflights and protected
archive checks. Array28695498_0..3 (096,229,192,221) and afterany publisher28695499
released after actual allocation/launcher/source validation:2CPUs/18GiB/2h each,
publisher1CPU/2GiB/2h,mit_normal,advanced account/QoS,no throttle. No original or
predecessor numerical operation repeated. Await fresh scientific/lifecycle PASS.

## Final Completion, 2026-11-11 15:52 UTC

All four numerical tasks28695498_0..3 COMPLETED0:0 in3m08s,2m06s,1m39s,1m40s.
Each reused its saved passing primary, ran exactly one new direct reference,
and passed complete flow/export/native-import/archive verification.76 saved
completed flow calls were reused; none repeated.096/229 each completed two
previously unstarted later calls;192/221 required no later calls. Scientific
checks unchanged. Slurm MaxRSS ranged1158952..1937892KiB, within18GiB requests.
Publisher28695499 COMPLETED0:0 in48s, published all four and closed their cases.
Each publication preserved1042 existing sibling artifacts and original evidence.
All four downloaded compact archives independently passed every-member checksum
verification locally. No retries or additional numerical jobs are needed.

Final bounded receipt audit PASS:2592/2592 frozen cases,1353024/1353024 positions,
no remaining cases/positions. Canonical522-position completion receipts match
the exact frozen catalog and plan seals; this is not a fresh full-campaign
archive rehash. All three campaign controller states inactive. Old campaign
and unrelated jobs untouched. See sibling phase1_completion_audit_20261111_v1/
COMPLETION_AUDIT.json and README.md locally and under the shared shaowen root.
Preserve historical failures; they no longer represent unresolved work.
