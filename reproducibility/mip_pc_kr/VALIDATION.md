# Preservation validation, 2026-09-28

The sources are copied, not modified in place. `SOURCE_MANIFEST.json` records
each original path, exact size and SHA256 and identifies pre-existing pins
verified during collection. The collector uses an explicit source allowlist;
it does not traverse campaign result trees or change jobs. Its script hash is
recorded in the manifest. Snapshot line endings are protected by Git attributes.

## Tests executed for this publication

Local environment: Windows x86-64, Julia 1.12.6, GNU C++/quadmath from Strawberry,
Python 3.10 with NumPy 2.2.6, SciPy 1.15.3 and h5py 3.14.0. The local bridge was
built from the unchanged archived C++ sources using the original build tool.
It is a fixture-test binary, not a replacement for the included Linux reference.

| Test suite | Passing checks |
| --- | ---: |
| Preserved MIP Pc tests, including independent paths and streaming | 746 |
| Preserved fixed100 pressure planner and binary round-trip | 72 |
| Preserved coupled pipeline, analytic flow and exact-state cache | 411 |
| Managed corrected-flow backend and native factor lifetimes | 15 |
| Package/source/input/overwrite/522-position Pc-only/export safeguards | 9 |
| Preserved physical table exporter | 10 |
| Preserved alternate Pc-only converter | 6 |

The first four suites total **1,244 Julia assertions**. The last three total
**25 Python tests**. The 522-position test uses synthetic mapped tables and
checks bitwise preservation of both Kr columns, saturation coordinates,
post-breakthrough Pc nodes and other scientific datasets, including sampling
identities. Deliberately changed identities and checksums are rejected.

The coupled fixture reproduces the qualified pipeline source identity:

```text
55e54e0ec7b5ebb179ab296cfa983baac8d8a40e0c1fbf1a2a32dd8415bba5f9
```

Re-run small fixtures from this package, with a fresh output directory:

```sh
python workflow.py verify
python -m unittest discover -s tests -p test_workflow.py -v
python -m unittest discover -s snapshot/historical_releases/mip_matrix_launch_20260911_v1/vendor -p 'test_*.py' -v
python tests/run_numerical.py /new/private/path/fixture-check --julia /path/to/julia-1.12.6
```

The numerical harness saves command lines, logs, log hashes, platform/version
and source-manifest identity in `PASS.json`. Retained publication evidence is
under `validation/`. None of these tests repeats accepted campaign calculations.

## Limits

These are source-preservation and small-fixture tests. They are **not** a new
full-case Linux campaign qualification, a fresh native reservoir import, a
reservoir simulation rerun, or a fresh checksum audit of all external datasets.
The full sparse worker's resource probe depends on Linux `/proc`; its complete
cluster execution was not rerun on Windows. Historical acceptance and source
locks are preserved separately from this publication's new test evidence.

Licensed MATLAB, recorded MRST/PREDICT versions, Julia runtimes, registered
packages and original sampling/checkpoint/geometry data remain necessary as
described in `REPRODUCE.md`. No original files, numerical checkpoints, accepted
results, controllers or running production jobs were changed or deleted.
