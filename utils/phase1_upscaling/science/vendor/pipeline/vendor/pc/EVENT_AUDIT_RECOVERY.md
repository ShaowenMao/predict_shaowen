# Core 3 endpoint-count diagnosis and validation-only recovery

The numerical module is unchanged byte-for-byte: src/, vendor/, Project.toml
and run_pc.jl match sealed release 938e58bbdf2c. No pressure sample, local Pc
property, event threshold, saturation, mobility cutoff or flow solver is changed.

## Root cause

At fixed output 100 (27024658.227243770 Pa), gridblock 12840's clay endpoint is
27024658.227243770 Pa in MATLAB and 27024658.227243774 Pa in Julia. The one-ULP
difference is 3.725290298461914e-9 Pa. Independent local-Pc preparation already
allows roundoff up to a relative 2e-14; actual endpoint differences are below
1.41e-15. However, the old validator inconsistently demanded identical interval
counts even for these rounded thresholds. Julia correctly counts 4415 native
events in the last interval; MATLAB correctly counts 4416. All pressure samples
are exactly equal, and neither engine omits an event from its own schedule.

This particular endpoint is inactive. The gridblock loses outlet connectivity
at 24174784.223862212 Pa in both models, below output 99 (25974789.581050597 Pa).
It is already invaded and persistently trapped in both models. Its frozen Sg
is 0.673520446622545, and krw is 2.9441224009257776e-7. The last three MATLAB
snapshots have identical Sg, trapped and invaded flags. A later local endpoint
cannot drain or change an already trapped gridblock in this algorithm.

## Validation policy correction, not numerical correction

Each engine's event count must match its own model EXACTLY. Event identity,
order and grouping must also agree. Cross-model bin differences are permitted
only for an adjacent-float endpoint of a gridblock proved already trapped before
the interval, with both path-loss thresholds below both endpoints and with
bitwise-frozen within-model Sg. Gas-entry shifts, active endpoint shifts, wider
roundoff, missing events, changed grouping, missing path proofs and changed
states are rejected. Every exception is recorded, and the result explicitly
states cross_model_event_bins_exact=false. We do not claim identical counts.

This supersedes only the old cross-model event-bin equality requirement for
proved inactive endpoints. All original numerical preparation, Sg, local Kr,
bulk, pressure, breakthrough, topology and zero-mobility gates remain unchanged.
No thresholds are snapped, no output pressure is moved and no MIP event is
skipped to make the comparison pass. The failed result is retained.

An additional Core 3 replay uses the frozen MATLAB local model in the unchanged
Julia engine, requiring bitwise-equal full states/local Kr and exact event counts
at every output. It isolates MIP correctness from cross-language local scaling.

Only Core 3 is rerun; its saved MATLAB reference and raw inputs are reused.
The five passing real-core calculations are not rerun. The recovery job verifies
their completion receipts and all reused input hashes before and after execution.
The old mismatch receipt's completed_points=100 counted the attempted final
point. New error reports distinguish attempted_point from fully verified points.

The module still exports raw physical Pc/Sg points, not approved reservoir
interpolation tables. No production changes or promotion are made.
