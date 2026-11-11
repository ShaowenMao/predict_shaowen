# Provenance And Preservation

## Immutable Identity

| Authority | SHA256 |
| --- | --- |
| Frozen configuration v2 LOCK | `7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36` |
| Scientific method | `ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832` |
| Production v5 LOCK | `c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf` |
| Round-two v4 LOCK | `988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019` |
| Round-three LOCK | `2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b` |
| Final-reference v2 LOCK | `8b61b7dabc42f57c858a2d137858fece866174ede388aa75fa6479137b2c0fbe` |
| Final completion-receipt digest | `df50b8b7695a7f2102b165998b46471f4a367ae183dc5fa04f9d879c91b52b4e` |

Full pins for all retained versions and every copied file are in
`SOURCE_MANIFEST.json`. Every runtime `LOCK.code` member is included. The
configuration's bulk data is explicitly listed as omitted, not claimed verified
by the offline checker. Original LOCKs are unchanged. Git attributes disable
line-ending rewriting for snapshots so their byte-level pins survive checkout.

The snapshot originates at `D:/codex_gom/studies/<package>/`; corresponding
cluster sources are `/orcd/data/juanes/001/shaowen/<package>/`. No source or
result in either original location was modified by packaging. The preserved
handoffs are dated historical records. Their "not yet adopted" or "not launched"
statements describe their own timestamp; later sections and final receipts
establish completion. Never edit a frozen LOCK to modernize those statements.

## Qualification Evidence To Follow

| Milestone | Recorded compute evidence |
| --- | --- |
| Pc40 + adaptive integration | Job 27293942, six representative runs, configuration README |
| v5 transition | 43 tests at 27415550; mixed old/new fixture 27415987/27416806; cutover 27416915 |
| Wider recovery window | 104 compute tests + saved-flow lifecycle at 28474289; transition 28475095 |
| Round three | 91 compute tests at 28591521; saved-output finish without solves before 382-core recovery |
| Targeted rolling batches | Corrected root-argument lifecycle at 28636221/28636410; adoption 28638567 |
| Precision batches | 83 checks at 28655933; all 38 canonical publications in its final handoff |
| Final four | 46 Python + 219 Julia checks at 28695227; workers 28695498_0..3; publisher 28695499 |
| Final catalog closeout | Saved 2026-11-11 completion audit: 2592 cases, 1,353,024 positions, no remainder |

These are preserved historical claims/evidence, not jobs rerun for this commit.
The local regression report in `VALIDATION.md` is separate from the historical
scientific qualification. Full diagnostic inputs, job logs, protected inventories
and publication records remain on shared disk. This bundle is a curated reusable
source reference, not a reconstruction of every failed historical candidate.

## Readable Scientific Source

`science/CachedAdaptiveCurve.jl` and `science/DrainageCache.jl` show the adopted
drainage-state reuse. `science/AdaptiveSelector.jl` is the pinned adaptive
selection rule. The original coupled solver/export stack is under
`science/vendor/`. Do not confuse the many historical launchers inside that
vendor tree with the Phase 1 production entrypoint in the v5 reference package.

Later precision/direct exceptions are in their own reference packages; they were
not silently inserted into the production bundle. Vendor licenses and attribution
for AMD.jl and LDLFactorizations.jl are retained in the final-reference package.
The repository's existing license applies to project-owned code; bundled
third-party code retains its own license. MATLAB and MRST installations are not
redistributed here.

## Updating This Reference

Do not edit a snapshot to make a test pass. Add a separately qualified successor
and document its scope instead. `capture_reference.py` is a maintainer-only
local importer with fixed source LOCK pins; it refuses an existing destination
and does not launch anything. The version-controlled manifest is the offline
integrity inventory, not a replacement for the original LOCKs or result receipts.
