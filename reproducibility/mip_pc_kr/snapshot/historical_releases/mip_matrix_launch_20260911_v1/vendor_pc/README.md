# FaultCorePc: standalone Julia candidate module

This recovery changes validation only. See EVENT_AUDIT_RECOVERY.md for the
Core 3 diagnosis and the explicitly audited inactive-endpoint count difference.
All numerical sources and the standalone CLI remain byte-for-byte unchanged.

This is a separate candidate, not a replacement of production. Starting from
existing fine-grid geometry/properties and calibrated reference tables, the
numerical module uses Julia standard libraries only. It does not invoke MATLAB,
MRST, JutulDarcy, external MEX kernels, phase-flow solvers, or reservoir simulators.
MATLAB scripts under validation/ are reference-data and interoperability tests,
not dependencies of the Julia numerical module.

## Scope

- Compact gridblock-specific local Pc curves: two fixed reference tables plus
  one per-gridblock scale. Sand uses the existing Leverett formula; clay uses
  the existing permeability/breakthrough correlation and fluid conversion.
- Permeability layouts 1, 3, or 6 columns retain the MRST fault-local convention.
  Kzz is column 1, 3, or 6, respectively; six columns mean xx, xy, xz, yy, yz, zz.
  Input permeability is m2, volumes m3, porosity dimensionless, pressure Pa.
- Existing scaling floors, contact angles, tensions, clay anchor, reference
  tables and endpoints are unchanged. The floors affect scaling only, never
  physical pore volume. Physical PV is gridblock volume times original porosity.
- The validated pure Julia event engine is vendored byte-for-byte unchanged.
  All gridblock entry and zero-krw endpoint events are retained. Sand Sg endpoint
  is 0.95, clay endpoint 0.70, and local wetting mobility means strictly krw > 0.
- The existing separated-event fast path is ported with its original guard:
  all entry pressures precede every closure pressure and the grid is connected.
  A pure Julia heap computes the same exact minimax/maximin threshold values
  as the MATLAB spanning-tree implementation. Path witnesses may differ on ties;
  only threshold values affect the physical states. No guard is widened.
- Otherwise the unmodified literal event engine runs, including persistent
  outlet-connected trapping. No events are removed to meet a time budget.

## Use without MATLAB

The exported raw.bin is a neutral input bundle, not a precomputed local-Pc model.
It contains original geometry, gridblock permeability/porosity/volumes, material
IDs, reference tables, settings, and the prescribed pressure schedule. Any
application can construct CoreInput directly instead of using this binary format.

```julia
include("src/FaultCorePc.jl")
using .FaultCorePc
input = read_input("raw.bin")
model, audit = prepare_model(input)
result = upscale(model)
```

For a streaming run with input/source/output fingerprints:

```text
julia --startup-file=no --math-mode=ieee run_pc.jl raw.bin NEW_OUTPUT_DIRECTORY
```

The CLI writes pc_curve.csv, states.bin and completion.toml. Existing output
directories are refused. A failed run has no completion receipt; the fixed-size
archive detects missing or extra records. No automatic resume or production
promotion is implemented in this initial qualification release.

## Sampling and saturation conventions

The six qualification cores retain exactly their accepted 100 pressure values.
fixed_from_dictionary reproduces the pressure-only nearest-log selection from
the supplied frozen pressure dictionary. The dictionary itself is an input;
generating a new campaign-wide pressure range is not silently introduced here.
The library accepts other strictly increasing schedules starting at zero, but
does not declare their resolution adequate.

pc_curve.csv stores PHYSICAL PV-weighted Sg, including pre-breakthrough gas,
matching the current candidate's raw fixed100 tables. It also stores invasion,
trapping, immobile-PV and breakthrough diagnostics. It does not replace Sg with
zero before breakthrough, collapse constant-Sg branches, extrapolate the initial
reservoir curve, or manufacture an entry-pressure anchor. Reservoir-ready
single-valued Pc(Sg) tables remain a separately reviewed export concern.
Existing finite-sampling accuracy flags are retained, not called execution
failures. No continuous interpolation-error bound is claimed.

## Kr interoperability

The streamed archive retains every output Sg vector, invaded/trapped flags,
breakthrough pressure and both local Kr fields, plus a hash of the entire raw
input. Julia and MATLAB streaming readers check its format and lengths.
The unchanged corrected-direct Kr solver can consume those fields with its
existing phase network; this module does not recompute upscaled Kr or intrinsic
permeability. The matching geometry and flow boundary metadata must be retained
by the caller. There is no approximate state-cache reuse across different cores.

## Qualification

See DESIGN.md. Local Julia unit tests, 24 independent MATLAB fixtures, and
cluster comparisons distinguish bitwise equality from harmless last-bit
differences in independently calculated local Pc parameters. Exact topology,
mobility-zero decisions and event counts are mandatory. The numerical gates are
declared before real-core execution and are never relaxed after a mismatch.
Use of pure Julia does not itself validate the physical upscaling assumptions.
