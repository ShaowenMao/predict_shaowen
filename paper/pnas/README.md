# PNAS manuscript and Supporting Information

This directory contains a PNAS-formatted alternative to the WRR manuscript. It is intentionally independent of the WRR entry points in the parent directory.

## Build

From PowerShell:

```powershell
cd D:\Github\predict_shaowen\paper\pnas
.\tools\build_pnas.ps1
```

Build the PNAS Supporting Information with:

```powershell
.\tools\build_pnas.ps1 -Document .\supporting_information.tex
```

The PDFs are written to `build/manuscript.pdf` and `build/supporting_information.pdf`.

The Figure S1--S6 generation commands, data provenance, and validation steps
are documented in [FIGURES.md](FIGURES.md).

For VS Code, open `pnas-paper.code-workspace` (or open this `pnas` directory
directly). The workspace recommends LaTeX Workshop, selects the **PNAS
document (pdfLaTeX)** recipe, builds on save, and refreshes the internal PDF
preview automatically.

## Journal-specific scientific narrative

- `manuscript.tex` contains the unchanged working title, PNAS metadata, abstract, Significance Statement, Figure 1, and section order.
- `supporting_information.tex` is the independently compiled PNAS SI entry point.
- `sections/introduction.tex`, `sections/results.tex`, and `sections/discussion.tex` form a PNAS-specific, question-led narrative. They do not reuse the workflow-led WRR Results structure.
- `sections/materials_and_methods.tex` is intentionally concise and retains only the study design and assumptions needed to interpret the results.
- The SI contains the complete procedures. `sections/si_full_fault_sampling.tex` and `sections/si_reservoir_analysis.tex` document the revised conditionally independent full-fault design and its analysis; PREDICT generation/convergence and replay/upscaling sections remain shared where the science is identical.
- Main figures, supplementary figures, tables, and `../references.bib` remain shared with the WRR project.
- The existing WRR files `../manuscript.tex` and `../supporting_information.tex` are unchanged.

The Results and final portions of the abstract and Significance Statement contain explicit `TODO` markers until the Phase-1 reservoir ensemble and targeted enrichment are analyzed. Quantitative findings must replace those markers before submission.

## Current study design represented in the PNAS draft

- 162 balanced, unweighted geologic scenarios.
- Six PREDICT throw-window libraries with 2,000 accepted realizations per geology-window pair.
- Phase 1: 12 independent full-distribution fields plus one representative medoid and two low/high stress benchmarks per geology.
- Only independent fields enter probabilistic and variance-based analyses.
- Targeted enrichment is selected after Phase 1 and is excluded from the balanced global decomposition.

## PNAS checklist

- Keep the abstract at or below 250 words.
- Keep the Significance Statement between 50 and 120 words and accessible to scientists outside the field.
- Organize Results around geologic controls, within-geology stochasticity, enriched conditional distributions, and representative-fault performance rather than software stages.
- Provide three to five keywords.
- Finalize author contributions, competing interests, corresponding-author email, acknowledgments, and data availability.
- Give sufficient methodological detail in the main manuscript even when extended diagnostics appear in SI.
- Submit SI as one separate PDF with independently numbered figures and tables.

See `TEMPLATE_SOURCE.md` for template provenance.
