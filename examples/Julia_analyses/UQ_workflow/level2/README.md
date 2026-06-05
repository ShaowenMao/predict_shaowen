# Level 2

Level 2 builds empirical within-window state objects from one 2000-run joint
permeability library per fixed window.

Distance convention:

- Local ranks and local normal scores are still saved and used for joint-rank-score
  ordering.
- The default clustering, medoid, and local perturbation-pool distance is
  physical log-unit distance in `log10(k)` space, where `a_c = 1` for all
  components.
- Use `--distance-metric local_normal` only as a legacy/sensitivity comparison.
- The default minimum cluster fraction is 5%, so a 2000-run window requires at
  least 100 samples for a non-singleton cluster.
- Low and high state libraries use cluster-aware target-mass selection: extreme
  clusters are included first, and only the needed part of the adjacent boundary
  cluster is added to reach the target `state_fraction`.
- Local perturbation pools are cluster-preserving: candidates are the
  intersection of the state library and the joint cluster containing the state
  medoid. The pool keeps the nearest
  `min(candidate_count, max(local_pool_min_count, ceil(local_pool_fraction * candidate_count)))`
  candidates.
- State-wide perturbation pools use the full low/high state library. This
  preserves broad within-state variability, including secondary modes.
- Violin width scaling defaults are user-facing inputs in the `[plotting]`
  section of `configs/level2_defaults.toml`. Command-line
  `--fixed-count-density-reference` can still override them for one plotting
  run.

Current Level 2 layout:

```text
workflow/
  run_level2_workflow.jl
  level2_workflow_config.toml
  steps/
    step01_load_window_library.jl
    ...
    step08_generate_level2_outputs.jl
```

Shared code lives in `lib/` and now follows the same focused-module style as
Level 3:

- `level2_ranks.jl`: local ranks, local normal scores, and joint rank scores.
- `level2_distances.jl`: physical log-space distance matrices.
- `level2_clustering.jl`: k-medoids clustering, silhouettes, and cluster order.
- `level2_state_libraries.jl`: low/high state libraries, medoids, and pools.
- `level2_state_object.jl`: assembles focused Step 2 outputs into the
  canonical saved Level 2 state-object schema.
- `level2_io.jl`, `level2_plotting.jl`, and `level2_selection.jl`: data I/O,
  visualization helpers, and sampling rules.
- `level2_output_*.jl`: table, figure, and sampling-output generators.
- `level2_validation*.jl`: optional holdout-repeat validation outputs.

The modular driver in `workflow/` is the main entry point. Workflow steps live
in `workflow/steps/`, while reusable table, figure, sampling, and validation
logic lives in `lib/`. This keeps the Level 2 folder consistent with Level 3:
`lib/` contains reusable functions and `workflow/steps/` contains the execution
sequence.

Recommended Level 2 execution flow:

Run the modular driver:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/workflow/run_level2_workflow.jl
```

The workflow config controls which outputs are generated:

```text
workflow/level2_workflow_config.toml
```

Use individual `lib/level2_output_*.jl` files only when rerunning one specific
output outside the full workflow:

- `lib/level2_output_marginal_histograms.jl` for raw marginal histogram
  screening.
- `lib/level2_output_tables.jl` for CSV summaries.
- `lib/level2_output_joint_clusters.jl` for the main Step 2.3 joint-cluster
  figures.
- `lib/level2_output_joint_clusters_3d.jl` for supplementary 3D joint-cluster
  views.
- `lib/level2_output_cluster_sensitivity.jl` for minimum-cluster-fraction
  sensitivity checks.
- `lib/level2_output_cluster_bootstrap.jl` for optional cluster-stability
  bootstrap checks.
- `lib/level2_output_state_libraries.jl` for low/high state-library violin
  figures.
- `lib/level2_output_perturbation_pools.jl` for local/state-wide
  perturbation-pool figures.
- `lib/level2_output_sampling_test.jl` for sampling tests or field-scale design
  preparation.
- `lib/level2_validation.jl` and `lib/level2_validation_figures.jl` for
  holdout-repeat validation.

Step 2.8 selection design CSV:

```text
case_id,window,mode,state_label,perturbation_pool
case_001,famp1,state,low,local
case_001,famp2,state,high,state_wide
case_002,famp1,independent,,
```

For `mode = state`, `state_label` must be `low` or `high`, and
`perturbation_pool` must be `local` or `state_wide`. For `mode = independent`, the
selector samples uniformly from the full 2000-realization window library.
Every selected row is one actual joint PREDICT realization, so
`kxx`, `kyy`, and `kzz` are never mixed componentwise.
