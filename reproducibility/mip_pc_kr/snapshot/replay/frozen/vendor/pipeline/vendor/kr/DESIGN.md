# Standalone Julia corrected Kr benchmark

## Scope

Core 7 is the fast pilot, Core 18 is an expensive case, and Core 3 exercises a
difficult shared-repair case. They reuse the completed corrected-solver study's
phase_inputs.mat packets, including fixed100 pressures, unique-state mapping,
local phase Kr fields, fine-grid network, and accepted intrinsic normalization.
There is no MIP rerun, reservoir simulation, JutulDarcy dependency, production
edit, or change to either running MIP pilot. Corrected direct remains the reference.

Both gas and brine dip-parallel curves are compared. Identical-state reuse is
inherited exactly; no approximate-state caching, cutoff, or sampling changes
are introduced. Julia prebuilds static adjacency once while preserving the
positive-face and boundary-connectivity rules on each call.

## Numerical equivalence

1. MATLAB exports the unchanged network and local Kr values. Each unique phase
   state and the absolute control exports a SHA256 fingerprint of kept nodes,
   ordered face indices, harmonic face coefficients, and boundary coefficients.
   Julia must match every fingerprint exactly before timing starts.
2. The original phase_precision.cpp and phase_precision_repair.cpp compile
   unchanged behind a minimal C-callable mxArray adapter. These retain binary128
   face sums, residuals, energy checks, operator application, and two-double
   pressure updates. Thirty-two cross-language tests require bitwise agreement
   with the original MATLAB MEX outputs, including difficult coefficient ranges.
3. Julia uses sparse LU (UMFPACK, two internal refinement steps explicitly set),
   followed by the same external correction loop and original binary128 audits.
   No claim of bitwise LU/pressure-factor agreement is made across library builds.
4. Audit or pressure-bound failures trigger the same paired shifted-Cholesky
   preconditioned, binary128-operator repair. The shifts affect only the
   preconditioner. Outer iteration limits, restart length, inner relative
   tolerance, both strict repair targets, and repair-sensitivity gates remain.
   IterativeSolvers 0.9.4 supplies GMRES with left preconditioning, restart up to
   50 and at most 8 restart-lengths of inner work. Its implementation differs
   from MATLAB GMRES; results must pass the same independent physical audits.
5. Every result must preserve zero-flow classification and agree with the
   accepted reference and paired MATLAB conductance within the existing 1e-5
   relative gate. Maximum observed differences are recorded, not rounded into
   a claim of exact floating-point identity. Full expanded 100-point Kr curves
   are checked using the unchanged accepted intrinsic normalization.

The legacy stop targets are residual <=1e-10, flux imbalance and energy error
<=1e-9, with at most eight corrections. Final legacy acceptance remains residual
<=1e-8 and flux/energy <=1e-5 plus pressure bounds. The paired repair requires
targets 1e-9 and 1e-11, and agreement <=1e-5. No target is relaxed to gain speed.

## Timings

Each core uses one 2-CPU, 18-GB, nonexclusive Engaging allocation. Three complete
paired repetitions alternate MATLAB/Julia order. The MATLAB worker is the
unchanged sealed solver benchmark; it calls corrected direct explicitly. Julia
warms the legacy and repair paths before timing. Both numerical timers include
network filtering, assembly, factorization, solve, corrections and audits.
MIP, startup, compilation, input/output, reference comparisons, and the separate
absolute-control time are excluded from the phase-curve total and reported
separately. Worker resource logs retain wall time and peak process memory.

Julia records stage timings. A separately profiled expensive MATLAB call gives
diagnostic function timings, not a benchmark speedup; profiler overhead is never
included in the three paired comparisons. Library/build differences are recorded.
The benchmark evaluates complete corrected implementations, not pure LU alone.

## Reproducibility and safety

The frozen numerical/solver studies and accepted packets are hash-verified
before and after use. All new artifacts are exclusive-created in independent
project-storage attempts. The compiled numerical source files are never edited.
No dense results use cluster scratch; disposable runtime caches use node-local
/tmp. A validated Core 7 completion gates the additional cores with afterok;
no job-array task limit or production hold is applied.

Vendored, unmodified Julia package sources and licenses are included and covered
by the release hash: IterativeSolvers 0.9.4, RecipesBase 1.3.4, PrecompileTools
1.2.1, and Preferences 1.5.2. No runtime package resolution is permitted. Engaging
Julia is pinned to 1.12.6; local controls use 1.11.6 and are repeated on Engaging.
The low-level bridge is compiled with GCC 12.2 and libquadmath on Engaging.

No solver is promoted based on incomplete, rejected, or incomparable timings.
Pilot success is required before the more expensive cohort proceeds.
