# Independent sampling for 324 geological-petrophysical scenarios

This is a new, isolated sampling package. Existing PREDICT libraries, old sampling,
and reservoir results are not modified. No reservoir or upscaling jobs are submitted.

- 162 existing base geologies crossed with atlas clay-Pc samples 4 and 6.
- 40-degree CO2-brine contact angle recorded for both clay uses.
- Eight independently sampled full-fault realizations per complete scenario.
- 324 scenarios, 2,592 cases, 522 assignments per case: 1,353,024 assignments.
- Uniform sampling with replacement from each existing 2,000-member window library.
- Joint kxx/kyy/kzz and exact PREDICT replay identities are retained together.
- Complete scenario identity, including Pc sample, enters the random stream key.
  The two Pc choices are NOT paired. Coincident library draws are allowed naturally.
- No old reservoir outcome is consulted. No stress or medoid cases are included.

## Deliverables

- `CONFIG.json`: sampling specification and source provenance.
- `phase1/scenario_catalog.csv`: all geological inputs and the added Pc factor.
- `phase1/case_catalog.csv`: all 2,592 cases, array positions, and physics status.
- `phase1/library_catalog.csv`: 972 source-window libraries and replay provenance.
- `phase1/manifests/*.csv.gz`: one complete segment-level manifest per scenario.
- `phase1/sampling_manifest_all_scenarios.csv.gz`: canonical combined manifest.
- `phase1/sampling_streams.csv`: independently verifiable deterministic streams.
- `phase1/sampled_fields.h5`: exact permeability/replay arrays; axes and units stored.
- `validation/`: source-integrity, seed, manifest, and field checks.

The 70-degree curve-data file cited in CONFIG is provenance only, not a solver
input. This sampling task does not convert curves or generate new Pc/Kr tables.
New 40-degree physics, Pc/Kr upscaling, reservoir-input export, and simulations
must be prepared and qualified separately. Existing numerical outputs must not
be relabeled as results for these new cases.

## Reproducibility

The original SHA-256 length-prefixed identity and unbiased uint64 rejection
sampling algorithm is retained, with a new scenario-specific version namespace
and base seed 20261029. Source-row indices and slice IDs are one-based. NumPy/HDF5
case positions are zero-based and explicitly mapped in the case catalog.

`code/generate_sampling.py` refuses to overwrite an existing `phase1` directory.
`code/test_sampling.py` checks the sampler against an original production vector.
`code/verify_draws.jl` independently reproduces every draw using the original Julia
implementation. Validation does not re-seed, reject difficult geological cases,
or select realizations based on anticipated leakage.
