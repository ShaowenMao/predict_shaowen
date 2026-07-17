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
     physical Kr bounds, the six Swi-medoid source rows, and exact agreement
     between permeability and Pc sample indices/replay seeds.
   - Write one compact reservoir-ready MAT file per geology/case and a QA CSV.

## Reservoir Pc Representation Options

The rigorous `full_slice` option remains the production reference. It keeps
all 522 native Pc curves and maps one dynamic-Kr shape per window to every
slice-specific saturation endpoint.

The optional `pe_branch_medoid` representation reduces nonlinear table
heterogeneity without changing permeability or porosity:

1. detect separated entry-pressure levels in each window using large gaps in
   `log10(Pe)`;
2. select one actual full-Pc-curve medoid inside each Pe branch;
3. assign each slice to the medoid of its own branch; and
4. map the window's validated normalized dynamic-Kr shape to each branch
   medoid endpoint.

Pe determines branch membership, while the complete Pc shape and native
`BulkSgMax` determine the medoid inside that branch. The full-slice MAT is
never overwritten. The reducer writes a separate MAT with compact branch
tables, 6-by-87 assignments, approximation diagnostics, and QA:

```matlab
outputs = build_pe_branch_medoid_reservoir_inputs( ...
    fullSliceMat, outputDir, ...
    'MinLog10PeGap', 1.0, ...
    'MinBranchCount', 2);
```

Run `run_case01_pe_branch_medoid_test.m` for the current Case 01 regression
test and publication-quality review figures.

For an integrated dynamic-Kr production run, select the output mode with:

```text
KR_DYN_RESERVOIR_PC_REPRESENTATION=full_slice
KR_DYN_RESERVOIR_PC_REPRESENTATION=pe_branch_medoid
KR_DYN_RESERVOIR_PC_REPRESENTATION=both
```

The default is `full_slice`. Branch mode still retains the rigorous
full-slice MAT as provenance and writes the reduced artifact separately.

Both representations now include an explicit `saturationRegions` block for
downstream MRST code. It contains a global 6-by-87 `SATNUM` map, a contiguous
region count, region and assignment tables, and direct lookup indices into the
stored Pc/Kr curve arrays. Global IDs combine window and local domain, so the
same local Pe-branch number in two windows never aliases one saturation region.

Use the accessor without performing any downstream clustering or detection:

```matlab
[SATNUM, pcRegionCurves, krRegionCurves, regionTable] = ...
    get_reservoir_saturation_regions(reservoirReady);
```

Legacy validated MAT files can be upgraded non-destructively with
`regenerate_reservoir_ready_with_saturation_regions.m`.

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
- `plot_upscaled_pc_curve_medoids.m`: optional full-Pc-curve medoid figures;
  these medoids are visualization diagnostics, not reservoir inputs.
- `build_pe_branch_medoid_reservoir_inputs.m`: optional branch-aware Pc/Kr
  simplification that preserves the rigorous full-slice artifact.
- `plot_pe_branch_medoid_reduction.m`: branch-medoid and along-strike
  assignment diagnostics.
- `run_case01_pe_branch_medoid_test.m`: reproducible Case 01 test driver.
- `build_saturation_region_metadata.m`: globally numbered SATNUM metadata
  shared by both Pc representations.
- `get_reservoir_saturation_regions.m`: direct downstream accessor returning
  SATNUM and one Pc/Kr curve pair per global saturation region.
- `regenerate_reservoir_ready_with_saturation_regions.m`: non-destructive
  upgrader for existing validated reservoir-ready MAT files.

The historical `median_examples` filenames identify the original development
examples. The production selection terminology is **Swi medoid**.

## Key Production Settings

```text
KR_DYN_SELECTION_MODE=swi_medoid
KR_DYN_PC_PRESTEP_MODE=precomputed
KR_DYN_LINEAR_SOLVER=amgcl_require
KR_DYN_1D_AD_SOLVER=robust
KR_DYN_EXPORT_RESERVOIR_READY=1
KR_DYN_RESERVOIR_PC_REPRESENTATION=full_slice
KR_DYN_PERMEABILITY_INPUT=<path>/texas_field_sampling_compact.mat
PC_IP_ENABLE_MEDOID_DIAGNOSTICS=0
```

Use `KR_DYN_SELECTION_MODE=all` only for full-87 validation benchmarks.

## Reservoir-Ready Output

Each `reservoir_ready_<geology>_caseNN.mat` contains:

- `windowLabels` and `sliceIndices`;
- an `effectivePermeability` block with 6-by-87-by-3 `kxx`, `kyy`, and `kzz`
  arrays in mD and m^2, ordered as window-by-slice-by-component;
- a 6-by-87 matrix of exact volume-weighted upscaled porosity;
- a 6-by-87 cell array of native Pc curves;
- a 6-by-87 cell array of endpoint-consistent Kr curves;
- an explicit `saturationRegions.SATNUM` map, global region definitions, cell
  assignments, and curve lookup indices;
- the six Swi-medoid selections; and
- selected PREDICT sample indices, exact replay seeds, state/pool labels, and
  input-file provenance.

For each window-slice fault cell, porosity preserves fine-grid pore volume:

```text
phi_upscaled = sum(phi_i * bulk_volume_i) / sum(bulk_volume_i)
```

Permeability, porosity, and Pc are tied to the same replayed PREDICT
realization. The exporter verifies this identity from the selected sample
index and exact replay seed before writing a case file. Porosity is not
represented by a medoid and requires no additional flow simulation.

The companion `reservoir_ready_qa_summary.csv` must report `Passed = 1`
before the artifact is used downstream.
