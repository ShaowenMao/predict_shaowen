# Reuse For Phase 2, Phase 3 Or Another Campaign

Reuse the architecture, not the historical launch records. Keep one small finite
handoff chain and the existing checkpoint/acceptance contract. A new general
manager, cross-campaign ownership database or global planning scan is unnecessary.

## Minimal Adaptation

1. Create a new isolated source/configuration/output root and a unique campaign
   identity. Keep all Phase 1 and older campaign directories read-only.
2. Freeze the new catalog and exact per-core assignments once. Validate unique
   keys, window/strike coverage, original library/seed identities and profile
   routing. Do not reuse the Phase 1 regex, 2592-case count, 522/87 mapping or
   324-scenario assumption unless the new design really has those dimensions.
3. Explicitly pin contact angle, host/fault-clay reference profiles, interpolation
   and endpoints, Pc pressure states, adaptive selector and numerical checks.
   A profile or physics change is a new scientific method/configuration, not a
   command-line override of a frozen source.
4. Start from v5's 32-core packets and rolling eight-case handoff logic. Carry
   forward the qualified allocation-aware time-budget/finalization guard from
   round two, plus its regression tests, rather than repeating v5's known timeout
   limitation. This combination itself needs new qualification before adoption.
5. Replace hardcoded roots and runtime identities in a new candidate only.
   Pin the actual interpreter versions, MATLAB/MRST/PREDICT source inventory,
   Julia environments, exporter, native importer and scientific bundle. Do not
   copy a historical LAUNCH_AUTHORIZATION, SCOPE, ACK, claim or cursor.
6. Generate a fresh explicit-path launcher. Slurm executes a spooled script, so
   `$0` is not a reliable source-directory locator. Verify worker arguments,
   output root and source pin through an actual held submission before release.

## Qualification Ladder

- Run the offline regression suite and the new configuration integrity check.
- On compute, check the real deployed source/runtime and representative input
  preparation, including every new Pc profile, largest grids and slow tails.
- Prove a real 32-core packet's flow/export/native/archive lifecycle. Interrupt
  at a controlled checkpoint and prove resume does not repeat accepted calls.
- Exercise a nine-case finite scheduling fixture: rolling refill, one deliberate
  failure, failed/timeout finalization, uncertain/lost ACK rejection, no duplicate
  submission and no accidental scientific credit from fixtures.
- Complete and verify one full real case, then expand directly to four cases if
  authorized, then the continuously replenished eight-case window. Measure
  throughput, memory and stage timings; do not promise a rate from another mix.
- Obtain explicit launch authorization bound to the new scope/source, submit one
  seed, and retain its intent/ACK. Do not bootstrap another control chain.

CPU jobs always explicitly use the advanced CPU account/QoS and `mit_normal`.
Start from measured 2-CPU/18-GiB needs, not a new allocation to chase occupancy;
measure new geology before changing requests. Let Slurm arbitrate shared CPU/RAM
with unrelated work. Keep an association submission-count guard if needed, but
do not treat unfamiliar unrelated job names as campaign failures.

## Recovery And Acceptance

Finish the broad first pass before chasing isolated difficult cores. Verify saved
complete outputs first, then recover only interrupted or proven-unstarted cores.
Use a wider recovery-only case window under a fixed worker-reservation ceiling.
Classify residual exceptions rather than running blanket retries. Requalify each
changed solver/export path through full native import and publication, not just
a passing diagnostic system. Preserve exact inputs and completed calculations.

## Dependencies Not In Git

The repository is deliberately not a full campaign backup. It omits large
per-core sampling manifests/HDF5, source PREDICT checkpoint libraries, prepared
pilot binaries, runtime depots, MATLAB/MRST installations and bulk results.
`SOURCE_MANIFEST.json` enumerates every omitted LOCK-listed configuration file.
Retrieve old evidence from its pinned original location if reproducing Phase 1;
generate and independently freeze new evidence for a new campaign.

Historical targeted scripts also refer to external held-core inventories, saved
system probes, prior study outputs and original publication scopes. Those are
data dependencies, not a portable exception list for the next phase. Preserve
the checks but build a fresh, disjoint, explicitly authorized recovery scope.

Python 3.12 is used on Engaging. The production flow runtime used Julia 1.12.6;
native import used Julia 1.10.4. Serialized Julia inputs are not guaranteed
portable across versions. Repointing an environment without qualification is not
a harmless deployment change. See original `worker.py`, executable/runtime pins
and the readable scientific `runtime.toml` for exact paths. The latter also pins
the external compiled `libphase_bridge.so`; its binary/runtime must be verified
on compute, not replaced by an arbitrary rebuild or silently omitted.
