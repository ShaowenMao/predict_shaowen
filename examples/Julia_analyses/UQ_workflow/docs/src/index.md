# UQ Workflow Documentation

This is a preliminary local documentation site for the Julia UQ workflow.

The current documentation focuses on **Level 2: within-window PREDICT
stochasticity**, because that is the part of the workflow currently implemented
and tested against the six-window proxy dataset.

## Current Focus

Level 2 converts each window's 2000 PREDICT realizations into:

- joint permeability clusters,
- local ranks and joint rank scores,
- low/high state libraries,
- representative state medoids,
- local and state-wide perturbation pools,
- state-conditioned and independent sampling pools.

## Main Entry Point

The modular workflow driver is:

```julia
examples/Julia_analyses/UQ_workflow/level2/workflow/run_level2_workflow.jl
```

Typical command:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/workflow/run_level2_workflow.jl
```

## Pages

- [Level 2 Workflow](level2_workflow.md): driver and step-by-step workflow functions.
- [Level 2 API](level2_api.md): shared algorithm, I/O, plotting, and sampling APIs.

