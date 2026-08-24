# Figure S5: directional permeability upscaling

This directory contains the self-contained workflow for PNAS Supporting
Information Figure S5. The upper panel renders three steady single-phase flow
experiments on the same W3 PREDICT fault-core realization. The lower panel
shows the marginal probability distributions of the linked fault-normal,
strike-parallel, and dip-parallel effective permeabilities from the matching
2,000-realization ensemble.

## Tracked inputs and provenance

The compact inputs required to reproduce the figure are committed in `data/`:

- `predict_w3_case14_sample590_collapsed_replay_compact.mat` contains the
  structured fine grid and clay-smear mask for displayed MATLAB sample 590;
- `predict_w3_case14_ensemble_2000.mat` contains the 2,000 linked permeability
  vectors in component order `(kxx, kyy, kzz)` and units of mD.

These data correspond to throw window W3 in the medium-sand, uniform-interbed
scenario with `zf=500 m`, `Vcl,s=0.2`, and `Vcl,c=0.5`. Their SHA-256 hashes
are:

```text
87c6e6b80c8f89b727fbcaa03c35b6a97dd8025bf8801ab690bad48d0193fa14  predict_w3_case14_sample590_collapsed_replay_compact.mat
7884b83fc24703f722408f91911e31e470798c2eb2ec962ba30f194f38764413  predict_w3_case14_ensemble_2000.mat
```

The renderer validates that the ensemble is a finite, positive `2000 x 3`
matrix. It records the selected sample, component order, boundary conditions,
render parameters, and source paths in the companion JSON output.

## Rebuild

From the repository root:

```powershell
python paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py
```

The script uses repository-relative defaults and writes:

```text
paper/pnas/figures/fig5_predict_directional_permeability_upscaling.pdf
paper/pnas/figures/fig5_predict_directional_permeability_upscaling.png
paper/pnas/figures/fig5_predict_directional_permeability_upscaling.json
```

The PDF preserves vector text, arrows, axes, and histogram geometry; the 3-D
fault-core renders are embedded at 600 dpi. Temporary PyVista renders are
stored in the ignored `_cache/` directory and may be reused with
`--reuse-raw`.

## Dependencies and validation

Install the packages in `paper/figures/workflow/requirements.txt` and ensure a
working LaTeX installation is available for Computer Modern text. Then run:

```powershell
python -m py_compile paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py
python paper\pnas\tools\figs5_directional_upscaling\render_directional_upscaling.py
paper\pnas\tools\build_pnas.ps1 -Document paper\pnas\supporting_information.tex
```

Visually inspect both the 600-dpi PNG and the Figure S5 page in the compiled SI
PDF before committing regenerated assets.
