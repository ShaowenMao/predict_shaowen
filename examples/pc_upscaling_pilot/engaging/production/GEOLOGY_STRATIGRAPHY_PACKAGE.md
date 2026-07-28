# Geology-Stratigraphy Companion Package

This post-QA stage creates one compact geology-specific stratigraphy MAT file
for every geology in a completed production run. It links the geology to its
ten validated full-slice fault-property cases without modifying the existing
Pc, Kr, permeability, or porosity outputs.

## Package Contents

The default output is:

```text
<run_root>/downstream_inputs/geology_stratigraphy/
├── geologies/
│   └── <geology_id>/
│       ├── geology_stratigraphy_<geology_id>.mat
│       ├── geology_stratigraphy_summary_<geology_id>.csv
│       └── geology_fault_case_links_<geology_id>.csv
├── geology_stratigraphy_manifest.csv
├── geology_fault_case_links.csv
├── SHA256SUMS
└── geology_stratigraphy.done.json
```

Each MAT file stores:

- the canonical geology ID and deterministic SHA-256 geology hash;
- original footwall and hanging-wall layer order, lithology, and thickness;
- the collapsed stratigraphy used by the production PREDICT calculation;
- six-window labels and relevant PREDICT configuration provenance;
- the fixed-grid reservoir mapping contract.

The downstream mapping uses the original layer sequence. It does not move
reservoir-grid boundaries, collapse adjacent reservoir cells, or change the
existing sand and clay property definitions.

## Safety and Validation

The stage runs only after the final case-completion gate passes. It requires:

- 162 unique geologies;
- six frozen PREDICT checkpoints per geology;
- cases 01 through 10 for every geology;
- one readable full-slice fault-property MAT file per case;
- exact geology and case identity matches in all 1,620 linked files.

Outputs are first written to an isolated partial directory. The package is
atomically published only after all counts, hashes, links, and checksums pass.
Re-running the submission validates and reuses a complete package instead of
rewriting it.

## Standalone Submission

```bash
RUN_ID=<run_id> \
RUN_ROOT=/path/to/production_runs/<run_id> \
FREEZE_ROOT=/path/to/production_freeze \
RUNTIME_REPO=/path/to/immutable/runtime_repo \
bash \
  examples/pc_upscaling_pilot/engaging/production/\
submit_geology_stratigraphy_package.sh submit
```

Use `plan` instead of `submit` to inspect the resolved inputs and output path.

## Integrated Production Workflow

Both production launchers submit this stage automatically after the final QA
gate:

- `submit_full_production_chain.sh`
- `submit_full_production_continuation.sh`

The continuation launcher also supports a package-only recovery: when all
scientific checkpoints are already complete, it validates or submits only the
missing stratigraphy package.

## Current Production Package

The validated 162-geology package generated for
`production_all1620_20260724_v1` is stored at:

```text
/orcd/data/juanes/001/shaowen/predict_shaowen/production_runs/
production_all1620_20260724_v1/downstream_inputs/geology_stratigraphy
```

It contains 162 unique geology MAT files, 1,620 unique full-slice case links,
and 488 checksummed package artifacts.
