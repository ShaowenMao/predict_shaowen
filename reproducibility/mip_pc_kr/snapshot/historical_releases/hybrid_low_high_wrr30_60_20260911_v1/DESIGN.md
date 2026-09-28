# Isolated low/high hybrid comparison

Four scenarios: s01_c018_case07 and s03_c003_case08, each at WRR30 and WRR60.
The intermediate hybrid jobs and every production artifact remain untouched.

- Reuse the exact original draw, replay seed, geometry, intrinsic permeability and porosity.
- Retain candidate cell-based MIP Pc, outlet-connected trapping with krw > 0,
  sand endpoint 0.95, fixed 100 pressure states plus connectivity events, and
  physical pre-breakthrough gas saturation.
- Reuse accepted Pc results. A missing coupled result may be recovered using
  only the identical frozen MIP engine, without the unnecessary MIP-Kr solves.
  An accepted control core is independently reproduced to absolute Sg error
  <= 1e-13 first. Never accept a partial coupled curve.
- Recompute six production-style dynamic Kr fits independently for every
  scenario, reselecting medoids from updated endpoints. Retain the production
  uniform-sand K/phi, capillarity-disabled flood and robust 1-D fitting method.
  Map the fitted shapes to all 522 updated endpoints.
- WRR60 changes the local fault-clay Pc and Kr assumptions together.
  Sand, host layers, reservoir fluid/PVT and geometry remain unchanged.
- Preserve native Pc and fitted Kr exactly using the union of their knots.
  Run all-table native constitutive checks, 14 smoke runs and reservoir preflight.
- Launch a fresh 210-report-step reservoir run only after successful validation.
  Match production solver/settings (8 CPUs, 18 GB, QoI v4), retaining all
  report states for later movies. Data and temporary files stay on project disk.
- Six window fits may run concurrently, with no JobArrayTaskLimit. Whole-array
  afterok dependencies gate downstream steps. High-case preparation uses
  afterany for existing upstream writers, followed by strict input validation.

All source files are SHA256 sealed. Each stage checks protected inputs.
Submission receipts distinguish queued preparation from an actual reservoir run.
