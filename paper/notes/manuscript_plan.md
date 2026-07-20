# Manuscript plan

## Working contribution

A reproducible end-to-end uncertainty-quantification workflow that propagates 162 geologic scenarios through stochastic fault-property prediction, hierarchical three-dimensional fault sampling, multiphase-property upscaling, and 1,620 field-scale CO2-storage simulations.

## Main-text organization

1. **Introduction:** GCS decision problem, fault uncertainty, computational gap, and contribution.
2. **Offshore Texas Field Case and Geologic Uncertainty:** field model, growth fault, throw-window representation, and 162-scenario design.
3. **End-to-End UQ Workflow:** PREDICT, within-window states, cross-window coupling cases, along-strike sampling, exact replay, and multiphase upscaling.
4. **Field-Scale Multiphase Simulation and Analysis:** injection model, 1,620-case ensemble, quantities of interest, and uncertainty analysis.
5. **Results:** geologic controls, hierarchical reduction, multiphase properties, migration and containment, and uncertainty contributions.
6. **Discussion:** storage decisions, value and limits of reduction, multiphase fault representation, limitations, and transferability.
7. **Conclusions:** quantitative findings and actionable implications.

## Proposed narrative

1. Faults can seal or transmit CO2, making their uncertain properties central to storage decisions.
2. Fault-property uncertainty is high dimensional and structured; component-wise or single-representative reductions can destroy important joint and multimodal behavior.
3. Hierarchical sampling preserves actual joint PREDICT realizations while reducing the space to 10 transparent cases per geology.
4. Window similarity groups impose candidate coupling assumptions without claiming observed cross-window correlation.
5. Exact replay and rigorous multiphase upscaling connect selected effective permeability cases back to fine-scale fault-core physics.
6. Field-scale simulations quantify retention, fault and top-seal entry, upward migration, phase partitioning, and the uncertainty sources controlling them.

## Candidate main figures

1. End-to-end three-level workflow and data flow.
2. Geologic scenario design and representative PREDICT marginal distributions.
3. All-window joint permeability clusters in raw log10(k) space.
4. Low/high state-library component distributions.
5. Similarity-group structure frequencies and co-grouping matrix.
6. Representative independent, fault-wide, and grouped 6-by-87 permeability fields.
7. Upscaled Pc branch structure and effective Swi maps.
8. Reservoir-scale quantities of interest and uncertainty decomposition.

## Candidate supporting figures and tables

- PREDICT convergence analysis.
- Minimum-cluster-fraction and bootstrap sensitivity.
- Full geology-input table.
- State-regime composition and medoid tables.
- Pairwise normalized energy-distance matrices.
- Additional fine-scale map validation for smear-placement rules.
- Pc/Kr robustness and endpoint-consistency diagnostics.

## Decisions still needed

- Whether the primary paper includes the cell-union smear algorithm as a central contribution or validation detail.
- Final reservoir quantities of interest.
- Main-text versus supporting-information allocation.
- Data and software archival repositories and DOIs.
- Final statistical approach for attributing outcome variability across the uncertainty hierarchy.

## AGU checks

- Abstract below 250 words.
- Three key points, each no more than 140 characters.
- Continuous line numbering and draft spacing.
- Figures cited near their first discussion and accessible without relying on color alone.
- Conflict-of-interest statement.
- Open Research statement with archived data/software citations.
- Publication-unit count monitored throughout drafting.
