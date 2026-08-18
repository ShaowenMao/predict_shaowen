# Figure S3: fault-domain discretization

This directory contains the reproducible workflow for PNAS Supporting
Information Figure S3. The figure uses the protected Step62 mesh and shows:

- the complete active 3-D main-fault domain, colored by host interval;
- the 87 along-strike segments present in every top-seal throw window; and
- the central y-z cross section through the six PREDICT throw windows W1-W6.

The accepted 3-D view is rendered at three-times vertical exaggeration. Its
geometry is rasterized because it contains 150,597 wedge cells. All panel
labels, coordinate rulers, tick labels, legend text, and the panel-(b)
cross-section mesh are Matplotlib artists and therefore remain vector content
in the PDF.

## Inputs and checks

The workflow reads these files from the protected Step62 candidate:

```text
nodes_coordinates.dat
t.mat
ucids_sc2_2D.mat
```

The code stops if the data do not satisfy the established Step62 contract:

- 87 along-strike segments spanning 45 km;
- W1-W6 cross-sectional cell counts of 62, 63, 62, 64, 62, and 57; and
- 1,731 fault triangles per cross section, extruded to 150,597 3-D cells.

By default, the scripts look for the grid in the sibling checkout
`mrst_predict_sim_grid_integration`. Use `--grid-dir` to select another
location containing the same protected Step62 inputs.

## Rebuild

From the repository root, first regenerate the clean high-resolution 3-D
source render when the geometry, colors, camera, or segmentation changes:

```powershell
python paper\pnas\tools\figs3_fault_domain\render_fault_domain_overview.py `
  --grid-dir D:\Github\mrst_predict_sim_grid_integration\setup_shaowen_resolution\grid_candidates\step_62_matched_upper_lower_transition
```

Then compose the publication figure:

```powershell
python paper\pnas\tools\figs3_fault_domain\compose_fault_domain_windows.py `
  --grid-dir D:\Github\mrst_predict_sim_grid_integration\setup_shaowen_resolution\grid_candidates\step_62_matched_upper_lower_transition
```

The final files are written to:

```text
paper/pnas/figures/fig3_fault_domain_discretization.pdf
paper/pnas/figures/fig3_fault_domain_discretization.png
```

`render_fault_grid_multiscale.py` contains the supporting central-grid and
cross-section rendering utilities used by the composer. The committed
`paper/pnas/figures/source/figs3_fault_overview_vector_base.png` permits
layout-only changes without rerunning the more expensive PyVista render.

## Dependencies and validation

Install the Python packages in `paper/figures/workflow/requirements.txt` and
ensure a LaTeX installation is available for Computer Modern text. Validate
the scripts and rebuild the SI with:

```powershell
Get-ChildItem paper\pnas\tools\figs3_fault_domain\*.py | ForEach-Object {
  python -m py_compile $_.FullName
}
python paper\pnas\tools\figs3_fault_domain\compose_fault_domain_windows.py
paper\pnas\tools\build_pnas.ps1 -Document paper\pnas\supporting_information.tex
```

Visually inspect the 600-dpi PNG and the compiled SI PDF at publication scale.
