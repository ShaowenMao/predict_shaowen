# Replay, Pc, and Dynamic Kr Upscaling

This folder contains the rigorous production workflow that converts selected
Level-3 PREDICT realizations into slice-specific capillary-pressure and
relative-permeability inputs for reservoir simulation.

## Production Workflow

1. **Replay exact PREDICT realizations**
   - Replay one selected realization for every geology/case/window/slice.
   - Verify replayed effective permeability against the stored PREDICT value.
   - Stop before upscaling if any replay is missing or outside tolerance.

2. **Upscale all Pc curves**
   - Run connectivity-based invasion percolation for all 87 slices and six
     windows, giving 522 native Pc curves per Level-3 case.
   - Preserve each curve's native `BulkSgMax` and
     `EffectiveSwi = 1 - BulkSgMax`.
   - Use every slice-specific Pc curve in reservoir simulation.

3. **Select six Swi-medoid Kr realizations**
   - For each window, select the observed effective-Swi value minimizing
     total absolute distance to the 87 values.
   - With 87 slices, this is the actual realization at the median Swi.
   - The selection is deterministic and remains tied to its replay seed,
     sample index, source row, and fine-scale map.

4. **Run rigorous dynamic Kr upscaling**
   - Run one full-3D dynamic Kr calculation per window.
   - Use robust timestep controls and require AMGCL for production 3D solves.
   - Save one normalized dynamic Kr shape per window.

5. **Restore slice-specific endpoints**
   - Map each window's normalized Kr shape to all 87 slices using each
     slice's own Pc-derived `BulkSgMax`.
   - This produces 522 physical-axis Kr curves whose Swi endpoints are
     exactly consistent with the corresponding Pc curves.

6. **Validate and package reservoir inputs**
   - Verify complete 6-by-87 coverage, Pc/Kr endpoint identity, monotonicity,
     physical Kr bounds, and the six Swi-medoid source rows.
   - Write one compact reservoir-ready MAT file per geology/case and a QA CSV.

## Pc Medoids

Full-Pc-curve medoids are optional diagnostics only. They show the best single
curve under the configured full-curve distance, but they are not used in the
reservoir model because one curve cannot represent the observed along-strike
Pc variability.

Enable the diagnostic with:

```matlab
setenv('PC_IP_ENABLE_MEDOID_DIAGNOSTICS', '1')
```

Production runs leave it disabled.

## Main Drivers

- `prepare_full87_replay_median_examples.m`: exact replay and verification.
- `run_pc_upscaling_ip_median_examples_full87.m`: all native Pc curves.
- `run_kr_upscaling_dyn_median_examples_full87.m`: full benchmark mode or
  reduced Swi-medoid dynamic Kr mode.
- `export_reservoir_ready_pc_kr_cases.m`: final QA and MAT packaging.
- `validate_pc_guided_kr_representatives.m`: reduced-vs-full Kr validation.

The historical `median_examples` filenames identify the original development
examples. The production selection terminology is **Swi medoid**.

## Key Production Settings

```text
KR_DYN_SELECTION_MODE=swi_medoid
KR_DYN_PC_PRESTEP_MODE=precomputed
KR_DYN_LINEAR_SOLVER=amgcl_require
KR_DYN_1D_AD_SOLVER=robust
KR_DYN_EXPORT_RESERVOIR_READY=1
PC_IP_ENABLE_MEDOID_DIAGNOSTICS=0
```

Use `KR_DYN_SELECTION_MODE=all` only for full-87 validation benchmarks.

## Reservoir-Ready Output

Each `reservoir_ready_<geology>_caseNN.mat` contains:

- `windowLabels` and `sliceIndices`;
- a 6-by-87 matrix of exact volume-weighted upscaled porosity;
- a 6-by-87 cell array of native Pc curves;
- a 6-by-87 cell array of endpoint-consistent Kr curves;
- the six Swi-medoid selections; and
- input-file provenance.

For each window-slice fault cell, porosity preserves fine-grid pore volume:

```text
phi_upscaled = sum(phi_i * bulk_volume_i) / sum(bulk_volume_i)
```

It is calculated from the same replayed PREDICT realization as permeability
and Pc. It is not represented by a medoid and requires no additional flow
simulation.

The companion `reservoir_ready_qa_summary.csv` must report `Passed = 1`
before the artifact is used downstream.
