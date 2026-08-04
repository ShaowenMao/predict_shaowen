# End-to-End Workflow Figure

This standalone TikZ project builds the manuscript overview figure for the
end-to-end uncertainty-quantification workflow. It links three stages:

1. local fault property modeling with PREDICT;
2. full-fault property sampling and multiphase-property upscaling; and
3. field-scale CO2 storage simulation with JutulDarcy.

The three scientific panels preserve the original 190 x 118.775 mm
composition. A separate margin below the panels contains the explanatory
caption. Figure-authored text uses a 9 pt body-text baseline, while the three
panel titles use 10.5 pt regular text.

## Quick build

From PowerShell:

```powershell
cd D:\Github\predict_shaowen\paper\figures\workflow
.\build_workflow.ps1
```

The normal build uses the committed scientific assets and therefore does not
require the original multi-gigabyte simulation directories. It produces:

- `workflow_overview.pdf`: vector manuscript figure;
- `workflow_overview_preview.png`: fixed-width review image with the PDF's
  native aspect ratio; and
- `build/`: temporary LaTeX files.

Use `-Clean` to remove build products first:

```powershell
.\build_workflow.ps1 -Clean
```

Use `-RefreshScientificAssets` only when the original PREDICT/JutulDarcy data
are available and the cached scientific graphics should be regenerated:

```powershell
.\build_workflow.ps1 -RefreshScientificAssets
```

## Dependencies

Required for a portable cached-asset build:

- Python 3;
- the packages in `requirements.txt`;
- a LaTeX installation providing `pdflatex`, TikZ, `newtxtext`, and
  `newtxmath`; and
- Poppler's `pdftoppm` for the PNG review render.

Install the Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

Regenerating the Panel 1 Pc/Kr source figure additionally requires Julia with
`CairoMakie` and `MAT`:

```julia
using Pkg
Pkg.add(["CairoMakie", "MAT"])
```

## Scientific-source overrides

The committed derived assets make the default build portable. To regenerate
from relocated source data, set these environment variables before building:

```powershell
$env:PREDICT_WORKFLOW_PANEL1_SOURCE = 'X:\path\reservoir_ready_case01.mat'
$env:PREDICT_WORKFLOW_PANEL2_SOURCE = 'X:\path\panel2_source_data'
.\build_workflow.ps1 -RefreshScientificAssets
```

Panel 3 regeneration has additional command-line inputs for the JutulDarcy
case files, renderer, and plotting presets. Run
`render_panel3_migration_cross_sections.py --help` and
`prepare_panel3_uq_assets.py --help` when those sources have moved. Complete
source provenance is recorded in `assets/SOURCES.md`.

## Project layout

- `workflow_overview.tex`: final text, panel placement, arrows, and caption;
- `workflow_style.tex`: typography, colors, and reusable TikZ styles;
- `build_workflow.ps1`: portable one-command build and optional refresh logic;
- `render_panel1_pc_kr_curves.jl`: original CairoMakie Pc/Kr renderer with
  manuscript-scale label placement;
- `prepare_left_panel_assets.py`: transparent Panel 1 asset preparation;
- `prepare_middle_panel_assets.py`: Panel 2 maps and W4 curve ensembles;
- `render_3d_fault_geometry.py`: field-fault geometry rendering;
- `render_3d_reservoir_active_inactive.py`: reservoir context rendering;
- `render_panel3_migration_cross_sections.py`: matched plume cross sections;
- `prepare_panel3_uq_assets.py`: migration and containment diagnostics;
- `assets/raw/`: portable source fault-core realizations;
- `assets/derived/`: cached scientific graphics embedded by TikZ; and
- `assets/SOURCES.md`: source provenance and scientific role of each asset.

Generated PDFs, previews, caches, LaTeX intermediates, and exploratory crop
tests are excluded by the local `.gitignore`.
