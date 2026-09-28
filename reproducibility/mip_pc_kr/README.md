# Reproducible MIP Pc/Kr workflows

Preservation release, 2026-09-28. This package publishes copies of the qualified
scientific sources, not a replacement controller or a new scientific method.
The original local and Engaging files, source locks, jobs and results remain
untouched. Nothing here authorizes deleting original data or restarting a campaign.

## Two Pc treatments

| Workflow | Meaning | Preserved implementation |
| --- | --- | --- |
| Physical pre-breakthrough | Retain physical pore-volume-weighted gas saturation before outlet breakthrough. Pc and both Kr phases use the same MIP microstates. | `snapshot/pipeline`, `snapshot/backend`, `snapshot/kernel`, `snapshot/campaign` |
| Zero-before-breakthrough Pc convention | Use archived breakthrough-pressure onset in the reservoir Pc table. Retain the existing saturation coordinates, both MIP Kr columns and all post-breakthrough Pc nodes. | `snapshot/pc_onset/prepare_inputs.py` and the historical comparison/export releases |

The second workflow is a **Pc table treatment**, not another flow solver, a hard
invasion switch, or permission to clamp reservoir-cell saturation. Both workflows
preserve the original draws. Pc entry pressure and whole-core breakthrough
pressure are different quantities and must not be substituted for one another.

The current WRR60 campaign contract is 60-degree clay Pc/Kr, sand endpoint 0.95,
clay endpoint 0.8, 100 saved Pc pressures and at most 10 selected Kr states, with
corrected direct flow, conservation/precision checks and managed factor release.
Historical full100 and WRR30/WRR60 comparison sources are retained separately;
the Pc100/Kr10 release must not be described as identical to full100 sampling.

## Contents and integrity

- `SOURCE_MANIFEST.json` maps every archived source file to its original path,
  size and SHA256; existing qualification pins are checked during collection.
- `snapshot/` contains byte-preserved numerical and preparation sources,
  vendored dependencies and licenses, precision-bridge source, the reference
  Linux bridge, exporters, native importer/runtime, original tests and records.
- `workflow.py` relocates input/output/configuration paths while calling the
  unchanged numerical kernel and table transformations. It never submits jobs.
- `REPRODUCE.md` documents environments, inputs, safe commands and limitations.
- `RESULT_LOCATIONS.md` identifies the reservoir comparisons on shared disk.
- `VALIDATION.md` distinguishes current fixture tests from historical production
  qualification. `tests/` exercises checksum, overwrite, identity and Pc guards.

From this directory:

```sh
python workflow.py verify
python -m unittest discover -s tests -v
```

Do not edit `snapshot/`, normalize its line endings, or write results into it:
the numerical source identity includes file bytes. Git attributes preserve the
original bytes. Historical launch/dispatch scripts retain their original paths
and are **archival references, not recommended entry points**. Do not invoke
their submission commands against an existing campaign.

## Reproducibility boundary

This is the complete **source/workflow preservation package**, not a GitHub copy
of all production datasets or licensed/installed software. Original sampling,
checkpoints, exact case/slot mappings, raw geometry and case inputs remain
separately protected on Engaging, with manifests and hashes recorded here.
MATLAB, MRST, Julia runtimes and some external packages must be provisioned as
documented. A Git checkout alone cannot reproduce a geological case without
those exact inputs. Never replace them with a newly sampled draw or only a seed.

The existing cluster's numerical safeguards remain authoritative. No claim is
made that local fixture tests requalify every geometry, prove cross-platform
bitwise equality, or validate a fresh full-case Linux deployment.
