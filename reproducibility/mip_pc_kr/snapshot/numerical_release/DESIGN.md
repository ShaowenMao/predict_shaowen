# WRR60 campaign numerical handoff

This continues the authorized twelve-source-case campaign wave, not production.
All 522 placements per case keep their exact original draws and source locks.
The completed replay packet is the sole input. The fixed-100 planner, paired
WRR60 clay Pc/Kr conversion, cell-local Pc, krw>0 outlet trapping, physical
pre-breakthrough saturation, sand endpoint 0.95, and corrected-direct
dip-parallel flow kernels are borrowed unchanged from qualified releases.

## Gates and useful work

1. Planner verifies replay receipt identities, source/draw/geometry correspondence,
   and six exact known input identities. It records unsupported inputs as holds.
2. Handoff qualification replans those six already-completed controls from the
   campaign packets. Planned raw/geometry bytes must match their existing WRR60
   inputs exactly. Existing numerical artifacts and resource/native audit are
   verified and reused; no repeated flow comparisons. Unit tests exercise
   duplicate prevention, capacity accounting, resource routing and failed gates.
3. Fresh-process numerical canaries are actual retained campaign results. Select
   at most two per size/material stratum, preferring the largest cell count and
   permeability contrast, then the opposite clay-fraction extreme. They test
   the real input-to-worker-to-export path. Native import compatibility remains
   separately visible; unknown endpoint exceptions are never silently relaxed.
4. The controller releases only strata whose canaries complete the numerical,
   memory and native gates. Other strata remain held without blocking passed
   strata. It refills within 500 CPUs and 1,984 GiB, including other jobs and
   pending reservations. No exclusive nodes, array task throttle, or retries.
5. Each new core is its own two-CPU allocation and fresh Julia process. Initial
   measured-envelope qualification requests are 4/6/10/16/24 GiB for mixed cores
   up to 30/60/90/120/150k cells; homogeneous requests 4/4/6/8/12 GiB. These are
   pilot requests, not universal guarantees. Every accepted result requires
   zero memory-limit events and max(1.3*peak, peak+1GiB) headroom. The exact
   Core518 result is reused with its qualified 22GiB receipt, not recomputed.
   Failed numerical/headroom checks preserve output and trigger scoped review.

## Boundaries

Only Cartesian 100-by-n-by-100 cores, at most 150k cells, and permeability
contrast within the six verified broader controls are admitted initially.
The separate known Core287 exceptional correction context stays held. Exact
identity, not similar geometry, is required for completed-result reuse.
The nineteen 160k-cell replay holds remain a separate resource qualification.

New jobs explicitly use mit_amf_advanced_cpu account AND QoS, mit_normal;
node1622 is excluded for this wave. No controller starts work that cannot fit
the remaining pre-maintenance window with the existing 30-minute margin.
No numerical work runs on login nodes. Files remain on project disk, not
scratch or the currently inaccessible flash allocation. Storage headroom is
checked without claiming a verified user quota or global throughput guarantee.

Completion is split into numerical acceptance, per-core export/native review,
and full-case export mapping. The controller never submits reservoir runs or
declares a case export-ready merely because 522 numerical receipts exist.
Full-case mapping/export/runtime checks follow when all placements resolve.
Maintenance or controller expiry preserves immutable claims for reconciliation;
unknown submissions are not blindly resubmitted. Historical failures stay intact.
# Exact Repeated-Core Reuse (September 14)

This revision uses `numerical_wave01_reuse_v1`, not the previous live output root.
The completed parent preflight and six accepted controls are immutable inputs.
The previous pending controller is held while this replacement is qualified.

- Each admitted placement keeps its case, slot, region, window, slice and replay receipt.
- Streamed fingerprints cover every fine-grid property, reference table, sampling
  input, geometry, cell ordering, transmissibility and boundary value. Only the
  validated core label is excluded. Unplanned geometry must have the zero link
  specified by frozen prepare_slot.m and both file hashes must match its accepted
  replay receipt; planned geometry must link to the actual raw hash. The link is
  replaced by the normalized raw hash only after this stage-specific check.
  No rounding is used and arbitrary nonzero unplanned links are rejected.
- The key also binds the frozen scientific kernels, external pins, WRR60 scenario,
  physical pre-breakthrough saturation, fixed100 sampling, dip direction and direct
  solver configuration. Similar draws or average properties never establish reuse.
- One owner is scheduled per unique key. Existing exact controls are preferred.
  Resource and geometry holds remain in force; reuse cannot bypass qualification.
- A paired test uses two independently replayed placements of the same core. Both
  full100 Pc/Kr curves must be bitwise equal, export tables byte-identical, and both
  native imports must pass before the unique-task controller can run.
- Member input hashes are rechecked before an owner's calculation. Publication
  binds each destination region to the shared table's hash, not the owner's label.
  `resolve_case` refuses full export without all522 placements and native passes.
- The scientific Julia kernels and exporter are byte-identical to the parent.
  Existing accuracy corrections and resource bands are unchanged. Indexing uses
  streaming reads in12 independent per-case audits (1CPU/2GiB each), then a
  2CPU/3GiB collector. Completed serial fingerprints are reused. Read-cache
  advice releases streamed input pages without modifying data. Large numerical
  allocations remain restricted to cores needing them.
- This qualifies reuse for the current admitted first wave. It does not remove the
  remaining336 holds or itself authorize a complete2430-case expansion.
