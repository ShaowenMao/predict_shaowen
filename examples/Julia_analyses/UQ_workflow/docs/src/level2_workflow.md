# Level 2 Workflow

```@meta
CurrentModule = Main
```

The modular Level 2 workflow separates the scientific steps into small Julia
files under `level2/workflow/steps/`. The driver reads a workflow config, runs
the enabled steps, and delegates review outputs to stable table, figure, and
sampling scripts.

## Driver

The main driver is:

```text
level2/workflow/run_level2_workflow.jl
```

It reads:

```text
level2/workflow/level2_workflow_config.toml
```

and runs the enabled workflow blocks:

1. input-data screening,
2. modular Level 2 object construction,
3. table export,
4. figure generation,
5. sampling smoke test.

## Scientific Steps

The Level 2 object builder is intentionally split into one file per scientific
step:

| Step file | Purpose |
|---|---|
| `step01_load_window_library.jl` | Load one window's 2000-run PREDICT library. |
| `step02_detect_joint_clusters.jl` | Detect joint permeability clusters in physical `log10(k)` space. |
| `step03_compute_local_ranks_and_joint_rank_scores.jl` | Compute local ranks, local normal scores, and joint rank scores. |
| `step04_build_low_high_state_libraries.jl` | Build low/high state libraries from ordered clusters. |
| `step05_choose_state_medoids.jl` | Select actual PREDICT realizations as low/high medoids. |
| `step06_build_perturbation_pools.jl` | Build local and state-wide perturbation pools. |
| `step07_save_level2_object.jl` | Save the complete Level 2 window-state object. |

## Review Wrappers

The review wrappers call the stable scripts that already generate tables,
figures, and sampling-test outputs:

| Review file | Purpose |
|---|---|
| `run_level2_tables.jl` | Export CSV summaries from saved Level 2 objects. |
| `run_level2_figures.jl` | Generate enabled review figures. |
| `run_level2_sampling_test.jl` | Run the default sampling smoke test. |

## Why Some Workflow Docs Are Manual

The workflow files are scripts rather than package modules. For this
preliminary site, the workflow page is therefore written as a readable methods
overview, while the reusable Level 2 modules are documented automatically on
the [Level 2 API](level2_api.md) page.
