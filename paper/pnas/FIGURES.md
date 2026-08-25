# PNAS figure-generation workflow

This document records the reproducible workflow for the PNAS field-case
figures. Final manuscript figures are written to `paper/pnas/figures` in PDF,
PNG, and SVG formats. The PDF files are the LaTeX inputs; the 600-dpi PNG files
are convenient review copies; and the SVG files preserve editable vector text
and annotations.

## Dependencies

Use Python 3 with the packages listed in
`paper/figures/workflow/requirements.txt`. A working LaTeX installation is
needed because the plotting scripts use Computer Modern through Matplotlib's
`text.usetex` option.

```powershell
python -m pip install -r paper\figures\workflow\requirements.txt
```

The scripts expect the protected Step62 grid in a sibling checkout named
`mrst_predict_sim_grid_integration` or `mrst_predict_sim_grid_dev`. An explicit
`--grid-vtu` argument can be supplied when the checkout is elsewhere.

## Figure S1: offshore Texas field setting and geologic model

Figure S1 is assembled by:

```powershell
python paper\pnas\tools\plot_fig1_field_case_model.py
```

The composition script:

- crops and reorders the regional map and cross section from
  `figures/source/salo_salgado_2025_field_case_source.jpeg`;
- uses the committed Step62 3-D model render
  `figures/source/step62_two_faults_full_domain_unannotated.png`;
- reads the exact Step62 active 2-D VTU for the right-hand geology view;
- validates the 24,886-triangle Step62 visualization mesh (distinct from the
  21,245-cell active reservoir-simulation footprint);
- adds the panel labels, dimensions, coordinate axes, faults, injector, and
  geology annotations; and
- exports `fig1_offshore_texas_field_case_model.{pdf,png,svg}`.

The regional source artwork is adapted from Saló-Salgado et al. (2025). The
two source panels are kept as one raster asset so their original geologic
content is not redrawn or reinterpreted.

### Optional regeneration of the Step62 3-D model render

The committed model render lets Figure S1 be rebuilt without a multi-gigabyte
3-D VTU. When a compatible Step62 VTU is available, the underlying renderer is
`paper/figures/workflow/render_3d_reservoir_active_inactive.py`. A typical
full-domain command is:

```powershell
$activeVtu = '<Step62 3-D VTU with current geology indicators>'
$gridDir = 'D:\Github\mrst_predict_sim_grid_integration\setup_shaowen_resolution\grid_candidates\step_62_matched_upper_lower_transition'
$faultNodes = 'D:\Github\mrst_predict_sim_grid_integration\setup_shaowen_resolution\fnodcoord.mat'

python paper\figures\workflow\render_3d_reservoir_active_inactive.py `
  --active-vtu $activeVtu `
  --raw-grid $gridDir `
  --fault-node-coordinates $faultNodes `
  --secondary-fault-trace paper\pnas\figures\source\step62_secondary_fault_trace.csv `
  --output paper\pnas\figures\source\step62_two_faults_full_domain_unannotated.png `
  --cutaway-y 0 `
  --delineate-full-fault `
  --vertical-exaggeration 1.5 `
  --top-surface-opacity 0.20 `
  --opaque-front-xz `
  --opaque-active-sides `
  --injector 22500 12816 2012 `
  --injector-point-size 12
```

The secondary-fault surface is a visualization-only reconstruction from actual
Step62 mesh vertices. Its anchors and limitations are documented in
`figures/source/step62_secondary_fault_trace_provenance.md`; it is not a
simulated uncertain fault-property domain.

## Figure S2: top-seal interbed scenarios

The manuscript version uses the exact Step62 cross-section geometry:

```powershell
python paper\pnas\tools\plot_fig2_real_stratigraphy.py
```

It reads the tracked scenario definitions in
`examples/thickness_scenario_designs.csv` and the verified sand proportions in
`examples/footwall_sand_ratio_by_thickness_scenario.csv`. All six panels use
the same mesh geometry and topology; only lithology assignments change. The
script exports
`fig2_top_seal_interbed_scenarios_real_geometry.{pdf,png,svg}`.

`plot_fig2_thickness_scenarios.py` contains the shared scenario parsing,
apparent-thickness values, and style definitions. It can also create a
geometry-independent schematic for diagnostic use, but that schematic is not
the manuscript Figure S2.

## Figure S3: 3-D fault domain and throw-window discretization

Figure S3 is generated from the exact protected Step62 cross-section and the
87-segment along-strike extrusion:

```powershell
python paper\pnas\tools\figs3_fault_domain\compose_fault_domain_windows.py
```

The workflow validates the W1-W6 cell counts, the 1,731-cell 2-D fault
footprint, and the resulting 150,597-cell 3-D fault domain. Panel (a) uses a
three-times vertically exaggerated, high-resolution PyVista geometry render;
its rulers, labels, and legend are added as vector Matplotlib artists. Panel
(b), including its triangular mesh and throw-window boundaries, is fully
vector in the PDF. See
`tools/figs3_fault_domain/README.md` for input provenance, source-render
regeneration, and detailed validation commands.

## Figure S4: geology-to-PREDICT fault properties

Figure S4 is generated from the exact Step62 cross-section geometry and a
compact, source-verified W3 PREDICT replay:

```powershell
python paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py
```

The workflow uses one internally consistent example from the nonuniform,
medium-sand scenario through the selected W3 throw-window geometry, sand and
clay-smear placement, dip-parallel permeability, and porosity. The compact MAT
input, realization identifiers, source hash, repository-relative defaults,
and validation procedure are documented in
`tools/figs4_predict_geology_fault_properties/README.md`.

## Figure S5: directional permeability upscaling

Figure S5 is generated from a compact W3 replay and its matching linked
2,000-realization permeability ensemble:

```powershell
python paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py
```

Panel (a) applies the three local directional flow experiments to one
representative fault-core architecture. Panel (b) shows the corresponding
marginal ensemble distributions while retaining the scientific interpretation
of each row as a joint three-component permeability vector. Both compact MAT
inputs, their SHA-256 hashes, scenario identifiers, validation rules, and the
exact rebuild command are documented in
`tools/figs5_directional_upscaling/README.md`.

## Figure S6: PREDICT ensemble convergence

Figure S6 and its companion summary tables are generated in MATLAB from the
tracked reference-floor convergence tables:

```powershell
matlab -batch "addpath('paper/tools'); generate_si_predict_convergence_summary"
```

The source tables are in
`examples/gom_reference_floor_cell_union_psmear_full/tables`. The generator
pools all six throw windows, three permeability components, and 30 repeats at
each tested ensemble size; normalizes each score by its matching reference
floor; and exports a vector PDF, a 600-dpi PNG, and three audit tables. The
figure is written to
`paper/supplement/figures/predict_convergence_reference_floor_summary.{pdf,png}`.

## Validation and manuscript build

Run syntax checks and regenerate all six figures before a release:

```powershell
python -m py_compile `
  paper\figures\workflow\render_3d_reservoir_active_inactive.py `
  paper\pnas\tools\plot_fig1_field_case_model.py `
  paper\pnas\tools\plot_fig2_real_stratigraphy.py `
  paper\pnas\tools\plot_fig2_thickness_scenarios.py `
  paper\pnas\tools\figs3_fault_domain\compose_fault_domain_windows.py `
  paper\pnas\tools\figs3_fault_domain\render_fault_domain_base.py `
  paper\pnas\tools\figs3_fault_domain\render_fault_domain_overview.py `
  paper\pnas\tools\figs3_fault_domain\render_fault_grid_multiscale.py `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_fine_properties.py `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_smear_placement.py `
  paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py

python paper\pnas\tools\plot_fig1_field_case_model.py
python paper\pnas\tools\plot_fig2_real_stratigraphy.py
python paper\pnas\tools\figs3_fault_domain\compose_fault_domain_windows.py
python paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py
python paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py
matlab -batch "addpath('paper/tools'); generate_si_predict_convergence_summary"

cd paper\pnas
.\tools\build_pnas.ps1 -Document .\supporting_information.tex
```

Visually inspect the regenerated PNGs and the compiled Supporting Information
PDF at publication scale before committing updated figure assets. Figure S6
requires MATLAB in addition to the Python and LaTeX dependencies listed above.
