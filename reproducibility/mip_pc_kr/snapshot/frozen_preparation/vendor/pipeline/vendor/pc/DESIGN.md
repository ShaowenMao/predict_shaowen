# Standalone Julia Pc module qualification

## Isolation and references

Production, sealed MATLAB numerical releases, the completed MIP pilot and the
ongoing Julia Kr benchmark remain unchanged. The original numerical package is
d536359d3a72acb9ff9ece6abcf2158fc26ca5ea732e0d337a04668a5694e0bf.
The original Julia MIP source release is
5d1e9a456f673e920eddc605f69e126a289b2def3f4daf055e1478f318ca992a.
vendor/StandaloneMIP.jl is copied unchanged, with SHA256
b3c35054eb87c0c69b869bc600e85911880efc336a22d8cb64e84e64ebfc94b0.
Source, reference, configuration, input, runtime and output hashes are recorded.

Validation uses cores 3, 7, 18, 22, 25 and 47 from the accepted fixed100 cohort.
Core7 is the pilot; the other five are released only on pilot success via an
afterok dependency. Each task requests 2 CPUs, 18 GB and six hours, with no
exclusive node or JobArrayTaskLimit. No production job is held or reconfigured.
Dense outputs stay in project storage. Only disposable runtime caches use
compute-node /tmp. No reservoir simulations or Kr flow solves are launched.

Core25's completed MATLAB/C++ full-sweep reference states are reused, after
validating their completion receipt and exact model/schedule match. Its expensive
MATLAB MIP sweep is not repeated. Other cores get one frozen MATLAB reference
MIP sweep with the existing guarded fast path, without phase solves. Comparison
to accepted fixed100 bulk summaries prevents two new ports validating only
against each other.

## Tests and numerical gates

1. Pure Julia tests cover layouts, invalid inputs, scaling floors versus physical
   PV, disconnected grids, the heap against independent edge relaxation, exact
   fast-path versus literal states, fixed sampling, streaming and overwrite guards.
2. Twenty-four deterministic MATLAB fixtures cover sand-only/clay-only/mixed
   materials, varying permeability/porosity, all layouts, zero-PV cells, floors,
   disconnected boundaries, separated and overlapping event ranges. Each has
   100 output states and both local Kr fields. Pilot repeats these controls on
   Engaging, not just the local workstation runtime.
3. Gridblock scale, entry and endpoint pressures must match the frozen MATLAB
   preparation to relative 2e-14. Physical PV uses the same raw data. Event
   grouping/counts, sampling and graph/boundary/material arrays must agree.
4. Every full state requires identical invasion/trapping masks, identical
   zero/nonzero local Kr classifications, identical breakthrough status, and
   relative breakthrough-pressure error <=2e-14. Absolute Sg error <=2e-12,
   local Kr <=1e-11 and physical PV-summary error <=2e-12. Actual maxima and
   bitwise-equality indicators are reported separately. These are cross-language
   numerical-equivalence gates, NOT the Pc/Kr interpolation-error tolerances.
5. The complete streamed output is read back in Julia and in MATLAB. Both local
   Kr fields must survive interchange bit-for-bit; summaries must match the CSV.
   A mismatched or incomplete archive is rejected, not partially accepted.

Local fixtures initially found maximum Sg difference 1.11e-16, local Kr 2.22e-16,
bulk Sg/PV summaries 2.78e-17, and zero breakthrough-pressure difference.
All topology/mobility-zero decisions matched; 21/24 state histories were bitwise
identical. Thus this complete preparation port must not be advertised as
universally bitwise identical even though the frozen-input MIP engine was.

## Timing and delivery limits

Preparation, path construction and MIP/observation times are recorded separately.
This is initially one complete qualification run per real core, not a repeated
throughput benchmark. The prior 1.59x Core25 MIP speedup is not extrapolated to
the full new module or all realizations. The native CLI wall timer includes
output writes and is explicitly labelled separately from numerical timers.

This module provides raw Pc-pressure/Sg results and reusable states. It does not
approve reservoir-ready curve interpolation, replace fault geometry generation,
generate new pressure dictionaries, switch the Kr solver, or launch a campaign.
Those interfaces and broader throughput remain later acceptance gates.
