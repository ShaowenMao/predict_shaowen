# Restricted recovery of the 29 unaccepted full-fault cores

This release is derived from fullfault_julia_6cd1be107753. It does not modify
that release, the original production campaign, or any accepted core outputs.

## Numerical change

The physical TPFA matrix, connectivity, conductances, boundary values, MIP,
fine-scale Pc/Kr laws, pressure sampling, and binary128 operator/audit kernels
are unchanged. The normal corrected UMFPACK path is unchanged.

The previous fallback verified two cold-start solves using diagonal shifts
1e-12 and 1e-10 in the approximate, diagonally scaled Cholesky preconditioner.
The shift was not part of the physical operator. On the diagnosed weakly
connected phase networks, the more heavily shifted secondary preconditioner
stagnated at its existing iteration limit, despite the primary solve passing.

The primary retains 1e-12 as its first choice. On strict nonconvergence or a
nonpositive preconditioner factor, it can try 1e-14 and 1e-13. The secondary
uses a different shift, preferring 1e-14 then 1e-13 then 1e-12. Both start from
the same prescribed constant pressure, independently. No criterion is relaxed:
the primary target is 1e-9, secondary 1e-11, with the existing mass, energy,
residual and pressure-bound checks, and a relative conductance agreement of
1e-5. Unresolved cases remain failures. Failed correction histories are logged.

## Reproducibility and safeguards

- The exact 29-slot allowlist is enforced by the worker, not only the scheduler.
- Qualified old receipts from both previous releases are hash-preserved.
- Completed replay preparations are hash-verified and reused.
- The newly planned binary inputs must match the original attempted inputs
  bit-for-bit, including pressure sampling and geometry.
- Every previously completed phase-flow call in the failed attempt is compared
  with the retry using the existing 1e-5 conductance agreement tolerance.
- Mathematical controls verify the unchanged operator, analytical conductance,
  strict dual-solve criteria, and bitwise direct-path equivalence.
- Compute-node qualification additionally exercises the frozen first failing
  states from slots 7 and 11, existing Pc/flow/pipeline controls, and lineage tests.
- Reservoir-export approval remains false. Repeated-Sg flags are not removed.

The five memory-only attempts are first retried with their old, sealed code
and 64 GB. This release is used only for cores still unaccepted after that
separate resource-only retry. All outputs remain in candidate project storage.
