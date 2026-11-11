# Integrated Julia Pc and Kr candidate

This separate candidate couples the validated Julia gridblock-Pc/MIP module to
the validated corrected-direct Julia phase-flow solver. It does not replace,
edit, submit, or rerun the existing production reservoir workflow.

## Numerical workflow

1. Read original gridblock permeability, porosity, volumes, sand/clay reference
   tables, connectivity, drainage boundaries and the accepted fixed100 pressures.
2. Prepare gridblock-specific Pc curves, using the existing sand Leverett and
   clay breakthrough-pressure scaling. No region arithmetic means are introduced.
3. Run the unchanged persistent outlet-connected trapping MIP with krw > 0,
   sand endpoint 0.95 and clay endpoint 0.70. Keep all internal events and the
   validated guarded fast path. Keep physical PV-weighted pre-breakthrough Sg.
4. Compute intrinsic dip-parallel conductance with unit local relative permeability.
5. At each output, use the Julia MIP state to evaluate local phase Kr fields;
   feed them directly into the unchanged TPFA phase-flow solver.
6. Divide phase conductance by the freshly computed Julia intrinsic conductance.
   This is the same effective/intrinsic permeability ratio; the denominator is
   not imported from a MATLAB benchmark result. Cross-language differences are
   tested under the existing corrected-solver agreement gate.

The primary flow solver is SuiteSparse UMFPACK sparse LU, with the existing
binary128 residual/flux/energy audits, corrections and difficult-case repair.
The C++ binary128 helper and all validated numerical modules are vendored
unchanged. This is a Julia-driven workflow, not a claim that every numerical
library is written in pure Julia. It requires neither MATLAB nor JutulDarcy
during numerical execution.

Only identical local brine-and-gas mobility fields can reuse a within-core
flow result. A digest lookup is followed by byte equality checks. There is no
cross-core caching, smoothing, clipping of accepted Kr, skipped MIP events,
coarser sampling or relaxed physical tolerance.

## Inputs and use

```text
julia --startup-file=no --math-mode=ieee run_core.jl raw.bin geometry.bin precision_library NEW_OUTPUT [--states]
```

Build the precision helper once using the unchanged build script:

```text
python -B vendor/kr/build_bridge.py vendor/numeric NEW_BRIDGE_DIRECTORY
```

The neutral geometry.bin contains only intrinsic-permeability-weighted half-face
TPFA transmissibilities, connectivity, dip-parallel boundary gridblocks, core
length and interface area. It is SHA256-bound to raw.bin, and the runtime checks
that both inputs have identical gridblock ordering and boundary sets. Geometry
generation and new pressure-dictionary generation remain upstream; they are not
ported in this integration. Test packaging reads the previously saved MAT files
using export_geometry.m, without recomputing properties or solving flow. This is
not a MATLAB call from the numerical driver.

Outputs include raw pc_kr_raw.csv, per-solve diagnostics, timing/provenance and
hashed completion receipts. Dense states are optional and off by default; the
qualification jobs enable them to validate every gridblock and the Pc-to-Kr
handoff. Numerical completion is distinct from independent validation approval.
Existing output directories are rejected. Partial outputs are retained, not
silently resumed or overwritten.

## Qualification and boundaries

Start with Core 7, then release cores 3,18,22,25,47 using afterok with no array
throttle. Each task uses 2 CPUs and 18 GB, writes dense data only to project
storage, and does not change production scheduling.

Compare all 100 complete local states with saved MATLAB references: exact
invasion/trapping masks and zero-mobility classifications, Sg error <= 2e-12,
local Kr error <= 1e-11, physical bulk error <= 2e-12, breakthrough relative
error <= 2e-14. Keep the audited one-ULP inactive endpoint-bin policy from the
Core 3 Pc recovery; active event shifts remain failures. Check fresh intrinsic
conductance and effective Kr against frozen accepted results with relative
agreement <= 1e-5 and exact zero-flow classification. Record actual errors.

The six-core study is an integration/equivalence qualification, not another
three-repeat timing benchmark or a physical-truth validation. Fixed100 sampling
accuracy flags are copied unchanged, not relabeled as execution failures.
There is no automatic reservoir-table export or production promotion. Reservoir
interpolation/export conventions, full-522 orchestration, bounded storage,
restart policy and a larger robustness cohort remain subsequent deployment work.
