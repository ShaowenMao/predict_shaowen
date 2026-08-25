# Figure S4: geology, fault-core architecture, and fine-scale properties

This directory contains the self-contained workflow for PNAS Supporting
Information Figure S4. The figure follows one internally consistent example
from the field-scale geology into PREDICT:

- Step62 scenario 05 (medium sand proportion with nonuniform interbed
  spacing) in panel (a);
- throw window W3 in panel (b); and
- the matching sand/clay-smear architecture, dip-parallel permeability, and
  porosity fields in panel (c).

The displayed realization uses `zf=50 m`, `Vcl,s=0.1`, `Vcl,c=0.4`, MATLAB
sample 12, and seed 530101740. It was selected because its sand-filled cells
have both higher porosity and higher dip-parallel permeability than its clay
smears, avoiding a visually counterintuitive example while retaining a
heterogeneous fault-core architecture.

## Tracked input and provenance

The compact replay needed for panel (c) is committed at:

```text
data/predict_w3_case01_medoid12_replay_compact.mat
```

It contains the structured PREDICT grid, clay-smear mask, six permeability
tensor components, and porosity for this realization. Its SHA-256 hash is:

```text
b05c4dea3d3e58abcc20ac1656f57d077df213268d37b0a4eca2714ea0a38130
```

Panels (a) and (b) also read the protected Step62 two-dimensional VTU and the
tracked scenario-definition CSV files. By default, the script looks for the
Step62 grid in the sibling checkout `mrst_predict_sim_grid_integration`, then
`mrst_predict_sim_grid_dev`; use `--grid` if it is stored elsewhere.

## Rebuild

From the repository root:

```powershell
python paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py
```

The script writes:

```text
paper/pnas/figures/fig4_predict_geology_fault_properties.pdf
paper/pnas/figures/fig4_predict_geology_fault_properties.png
paper/pnas/figures/fig4_predict_geology_fault_properties.json
```

Panels (a) and (b), all annotations, the coordinate triad, and both color bars
are vector content in the PDF. The four detailed three-dimensional fault-core
views are flat-lit 600-dpi PyVista renders embedded in the vector composition.
Temporary source renders are written to the ignored `_cache/` directory; pass
`--reuse-raw` to reuse them for layout-only changes.

## Dependencies and validation

Install the packages listed in `paper/figures/workflow/requirements.txt` and
ensure that a LaTeX installation is available for Computer Modern text. Then
run:

```powershell
python -m py_compile `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_fine_properties.py `
  paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_smear_placement.py
python paper\pnas\tools\figs4_predict_geology_fault_properties\render_predict_geology_fault_properties.py
```

The renderer checks the input dimensions and property arrays, requires finite
positive permeability, and records the scenario, realization, geometry,
property ranges, and source paths in the companion JSON file. Visually inspect
the 600-dpi PNG and the Figure S4 page in the compiled SI PDF before release.
