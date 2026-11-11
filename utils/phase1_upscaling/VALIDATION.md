# Validation

## Safe Local Checks

```powershell
python -B utils/phase1_upscaling/verify_package.py
python -B utils/phase1_upscaling/run_offline_tests.py
```

The verifier checks snapshot hashes, original LOCK identities and complete code
coverage, pinned plan fixtures, all bundled scientific source members, the
readable source mirror, Python syntax and the saved completion-audit identity.
It does not hash any external libraries or production archive payloads.

The offline suites cover packet partitioning, rolling refill, completed-case
skipping, immutable archive/receipt protection, no blind resubmission, exact held
allocation checks, source-preserving cache transition, recovery-only wider
admission, allocation-aware finalization, child-process terminality, raw Slurm
accounting, saved primary identity and final case coverage. Each historical
module namespace runs in a separate process with a temporary run root.

Two process-group tests require Linux and skip on Windows. They were previously
qualified on compute; a Windows pass does not replace that check. Numerical
Julia/MATLAB work, full saved-result fixtures, source generation, publication and
launch CLIs are deliberately not part of the local test command.

Optional exact-spacing export regression (requires NumPy):

```powershell
Push-Location utils/phase1_upscaling/reference/phase1_exact_spacing_recovery_20261110_v1
python -B -m unittest test_exact_export -v
Pop-Location
```

No test in these commands recomputes a physical upscaling result. Historical
scientific/runtime qualification is documented in [provenance](PROVENANCE.md).
For a future phase, follow [the new-campaign qualification ladder](NEXT_CAMPAIGN.md).

## Packaging Check Record

Verified on Windows on 2026-11-11 with Python 3.14.5 and NumPy 2.4.4:

| Check | Result |
| --- | --- |
| Snapshot integrity | 796 files, 13 locked source packages; 992 bulk configuration files explicitly external |
| Packaging regressions | 8 passed |
| v5 first-pass/cache/transition | 43 passed |
| Recovery/allocation/timeout | 61 passed; 2 Linux-only process-group tests skipped |
| Final-reference accounting/primary proofs | 14 passed |
| Completion audit guards | 6 passed |
| Optional exact-spacing export | 8 passed |
| Julia adaptive selector, local Julia 1.11.6 | 26 passed; saved curves and synthetic values, no flow solves |

Total Python: **140 passed, 2 explicitly skipped**. The normal offline command
contains 132 passing tests and two skips; the extra eight require NumPy. This is
not a new qualification of the production Julia 1.12.6 solver or native importer.

Initial packaging checks caught a verifier field-name typo and an omitted
predecessor source used by the final-reference identity test. Both packaging
issues were fixed without modifying any preserved source or production file.
No new cluster job was submitted and no accepted result was recalculated.

All 796 manifest paths were present in the Git index. A fresh index checkout
with `core.autocrlf=true` passed the same integrity check and all normal offline
suites, confirming that Git preserves pinned source bytes on Windows too.
