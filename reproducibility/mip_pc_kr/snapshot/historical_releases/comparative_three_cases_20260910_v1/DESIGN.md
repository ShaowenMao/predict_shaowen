# Three-case paired comparison

## Question and scope

Compare candidate fault-core closures across the original low (`s01_c018_case07`, task 262),
intermediate (`s03_c022_case12`, task 1137), and high (`s03_c003_case08`, task 848) cases.
These labels describe the original selection, not a guaranteed ranking of new outcomes.
No production code, input, result, queued job, or running job is modified.

The factorial design has two WRR fault-clay wettability scenarios (30 and 60 degrees)
and two reservoir Pc conventions (physical prebreakthrough saturation retained, or
zero-before-breakthrough convention). There are 12 candidate reservoir inputs and
three existing production baselines, but only six distinct 522-core upscalings.

Within each case/angle pair, reuse the identical MIP states and identical Kr(Sg).
The breakthrough convention changes only the reservoir Pc table below the first
post-breakthrough saturation using the already-tested continuous export policy.
It does not rerun MIP with different physics or modify Kr abscissae, endpoints,
or constitutive values. It is not a new hard gas-invasion switch in the reservoir solver.

## Unchanged and changed inputs

Every core uses its exact production sample, seed, checkpoint, geometry, cell K and
porosity. Canonical upscaled Kxx/Kyy/Kzz and porosity are copied unchanged. Cell-based
Pc, strict-positive-krw outlet trapping, sand endpoint 0.95, fixed100 pressure sampling,
dip-parallel corrected direct phase-flow solves, and the frozen Julia numerical source
are shared. The numerical implementation is not edited by this orchestration.

The WRR60 scenario reuses the validated transform: clay Pc conversion at 60 degrees,
clay Swi 0.20, gas endpoint 0.80, Corey exponents nw=3.5 and ng=2.5. It is a paired
wettability scenario, not an angle-only experiment. Host/interbed properties remain
unchanged. The 30/60 pair uses exactly the same 100 pressure nodes. Range inadequacy
is reported, never silently extended. Numerical precision corrections are unchanged.

## Execution and reuse

1. Seal the wrapper source; pin production manifests, hashes and selected identities.
2. Verify all 522 existing intermediate30 and 522 intermediate60 accepted records on
   a compute node. Recover only intermediate60 raw assembly in this new study.
   Its previous assembly failed during MATLAB CEF startup (exit137), not during an
   identified numerical calculation. Pure data assembly uses no JVM; replay keeps
   the JVM because its architecture hashes require Java.
3. Replay each missing low/high core once. Compute and accept30, then transform and
   compute60 independently. Successful results/preparations survive the other angle's
   failure. There is no approximate cross-core reuse.
4. Pilot each of six window medoids plus minimum/maximum canonical Kzz per case.
   Verify outputs, identity, intrinsic normalization and storage before remaining cores.
5. Assemble raw curves, retaining every review flag. Repeated saturation nodes and
   finite-sampling errors are not automatically numerical failures or export approval.
6. Build both Pc exports using the separately validated exporter. Check all522 tables,
   exact Kr equality within convention pairs, source draws, K/porosity and native
   simulator import, initial state, derivatives, and matched production settings.
   A trapping-aware breakthrough extraction is required if the existing shortcut's
   assumptions do not hold; never silently substitute an unverified threshold.
7. Reuse the two running intermediate30 reservoir comparisons (22438751_1137 and
   22474505_1137). Submit remaining reservoir cases only after the export/native gates.
   No automated reservoir promotion is built into the upscaling completion step.

All new output, replay and temporary files are on project storage, not scratch.
Compute tasks request 2 CPUs and64GiB with single-thread-group BLAS/Julia limits of2.
Use whole-array afterok dependencies and no array-percent JobArrayTaskLimit. New
analysis excludes node1614 because its observed load makes wall-time comparisons
confounded; existing jobs remain untouched. Record actual node and resource timing.
Do not interpret cross-node wall-clock differences as a Pc-convention effect.

## Comparisons to report

First compare six-window distributions of native Pc/Kr curves and endpoints against
the original properties. Distinguish physical method differences from numerical error.
Then compare aligned reservoir time series through1000years: injection/BHP and regional
pressure, fault and overburden CO2 inventory (free/dissolved), leakage onset and cumulative
CO2/brine interface transport. Use corrected state/mobility-aware extractors only.
Check component mass balance and injection totals. Report nonlinear/linear iterations,
accepted/rejected timesteps, simulated-time progress and CPU/wall time separately.
Incomplete runs must not be compared as though they have equal physical horizons.

Paired contrasts isolate the Pc export convention at a fixed angle, or the combined
WRR wettability scenario at a fixed convention. Production-to-candidate contrasts
combine several closure changes and are not an isolated test of one mechanism.
These are sensitivity/comparison results, not independent proof of physical accuracy.
