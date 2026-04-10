# UQ For Smart Sampling

This folder collects Julia-based uncertainty quantification analyses that
post-process the MATLAB/PREDICT simulation outputs.

Current structure:

- `level2/`
  Level 2 uncertainty analyses for fixed geology, including reference
  agreement checks, pooled componentwise nonparametric objects, and
  exploratory reference uncertainty visualization.
- `level1/`
  Placeholder for future Level 1 uncertainty analyses across geology.
- `relative_contribution/`
  Placeholder for future workflows that compare or decompose Level 1 and
  Level 2 contributions.
- `common/`
  Placeholder for shared Julia utilities once multiple scripts reuse the
  same loaders, metrics, or plotting helpers.

The current Level 2 scripts are:

- `predict_reference_componentwise_agreement.jl`
- `predict_reference_componentwise_core_objects.jl`
- `predict_reference_uncertainty.jl`
