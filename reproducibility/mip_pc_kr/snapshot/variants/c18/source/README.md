# Isolated candidate reservoir comparison

Case: s03_c022_case12, original campaign task 1137. This is a trial, not
production promotion. No original production source, input, result, or scheduler
configuration is edited.

## Matched experiment

Reuse the completed production run as baseline. Run the candidate from t=0 to
1000 years with the identical 210-step schedule (injection ends at 50 years),
common MAT file, geometry/draws, intrinsic permeability, porosity, wells, initial
state, host/non-PREDICT properties, hysteresis, solver and tolerances. Invoke the
original immutable ed81af8e simulation entry point and locked-physics function
with Julia 1.10.4 and eight CPUs. The full run has the same 18 GB memory request;
its 48-hour allocation is a scheduling allowance, not a simulated-time change.

The only constitutive changes are the 522 candidate fault Pc/Kr tables and their
associated saturation endpoints. The candidate remains the tested 30-degree
scenario. The proposed future 60-degree scenario is NOT applied.

## Checks and isolation

- Checksums pin original/candidate inputs, runtime Manifest, validation receipts,
  the launch configuration, and the source bundle.
- `pair_input_audit.jl` compares every specific-input field. Only explicit table,
  endpoint and provenance differences are allowed; everything else must match.
- The full-grid preflight retains original production geometry, tensor,
  pressure, schedule, hysteresis and QoI checks. It also checks that assembly
  preserves every candidate fault table exactly.
- Only entry-plateau bookkeeping differs in the copied validation contract:
  eight unchanged base tables need adjustment; 522 candidate fault tables already
  start at their positive entry Pc. No simulator/importer behavior is changed.
- A case-only resolver avoids hashing all 2,430 production input files. Its
  manifest is explicitly labeled `isolated_case_pair_v1`, not a new production
  campaign. The sole allowed Slurm array task remains 1137 for traceability.
- Preflight and full jobs use `afterok`, without holding or changing production
  jobs. No archive/replacement controllers are submitted.
- All trial outputs, checkpoints, logs and TMPDIR live in the project-storage
  study root. Existing immutable runtime code on scratch is read only.
- Same QoI modes, retained years 25/50/100/1000 and two rolling checkpoints.
  No dense state export and no old restart reuse.

## Reproduction

Stage with Python 3.12 on Engaging, giving `prepare_study.py` the exact original
campaign manifest, immutable runtime, validated candidate export directory and a
new study root. Preparation refuses an existing root. Set `STUDY_ROOT` and
`JUTULDARCY_COMBINED_REPO`, then run `source/submit_pair.sh`. It submits only the
preflight and dependent full case, and records both IDs in `SUBMITTED.txt`.

Source files bearing original production names are isolated launch/check copies.
The unchanged full simulation still runs through the pinned original
`scripts/engaging/gom_sampling_cases.sbatch`. The production resolver is used only
for read-only schema normalization during staging, not for launching other tasks.

## Comparison after completion

First assess completion, timestep cuts/Newton iterations, runtime, saturation
bounds and mass balance. Then compare injector BHP, regional pressure, CO2
inventories/leakage, fault entry/exit component fluxes and the retained spatial
states at matched times. Numerical completion does not establish which closure
is physically most accurate. These curves differ jointly in fine-Pc assignment,
MIP trapping/reporting and Kr closure; this is not a one-factor attribution test.
