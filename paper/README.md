# Water Resources Research manuscript and Supporting Information

This directory contains the manuscript and Supporting Information sources for the end-to-end geologic carbon storage uncertainty-quantification workflow. Both documents use AGU's official `agujournal2025` LaTeX class and are configured for editing and PDF preview in VS Code.

## Quick start in VS Code

1. Reload VS Code after pulling or creating this project so the repository-level LaTeX Workshop settings are active.
2. Open `paper/manuscript.tex` or `paper/supporting_information.tex`.
3. Install the recommended **LaTeX Workshop** extension when VS Code prompts.
4. Select the recipe **AGU document (pdfLaTeX)** if prompted.
5. Save the file or run **LaTeX Workshop: Build LaTeX project**.
6. Open `paper/build/manuscript.pdf` or `paper/build/supporting_information.pdf` from the LaTeX Workshop PDF viewer.

From PowerShell, build with:

```powershell
cd D:\Github\predict_shaowen\paper
.\tools\build_manuscript.ps1
```

Build the Supporting Information with:

```powershell
.\tools\build_manuscript.ps1 -Document .\supporting_information.tex
```

Clean generated files with:

```powershell
.\tools\build_manuscript.ps1 -Clean
```

The shared wrapper accepts either root document, prefers the current per-user MiKTeX installation, and falls back to `latexmk` on `PATH`.

## Project layout

- `manuscript.tex`: title page, AGU metadata, abstract, and section assembly.
- `supporting_information.tex`: independent WRR Supporting Information root document.
- `sections/`: manuscript text split by major section.
- `supplement/`: supporting sections, figures, tables, and SI-specific guidance.
- `figures/`: publication figures used by the manuscript.
- `tables/`: standalone table fragments when needed.
- `references.bib`: BibTeX database.
- `notes/manuscript_plan.md`: scope, claims, figure plan, and writing checklist.
- `tools/build_manuscript.ps1`: portable build and clean wrapper.
- `agujournal2025.cls` and companion files: vendored official AGU template.

## Template provenance

The AGU files were copied from the official `AGU-Publications/agujournal2025-latex-template` repository at commit `355052226d872cf6b9211c12b73b2ed2da133a7d` on 19 July 2026. See `TEMPLATE_SOURCE.md`.

## Writing conventions

- Keep conclusions traceable to a figure, table, or archived analysis output.
- Use `\todo{...}` for unresolved text rather than silently assuming a result.
- Preserve each selected permeability realization as a joint `(kxx, kyy, kzz)` vector.
- Use **cluster**, **state**, **local pool**, **state-wide pool**, and **window similarity group** consistently.
- Add data and software repository DOIs before submission.

## Supporting Information conventions

- Supporting text, figures, tables, and captions belong in one independently compiled PDF.
- Figures, tables, equations, and numbered sections use the `S` prefix automatically.
- Keep central analysis and interpretation in the main manuscript; use the supplement for methods and diagnostics.
- Do not package datasets or software in the supplement. Deposit them and cite them through the Open Research Statement.
- References used in the supplement must also be included in the main manuscript under `References From the Supporting Information`.
