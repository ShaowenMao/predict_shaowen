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
  review/

scripts/
  00_screen_input_data/
    plot_marginal_hist_screening.jl

  02_export_tables/
    export_level2_tables.jl

  03_review_joint_clusters/
    plot_joint_clusters.jl
    plot_joint_clusters_3d.jl
    run_joint_cluster_sensitivity.jl
    run_joint_cluster_bootstrap.jl

  04_review_state_libraries/
    plot_state_library_distributions.jl

  05_review_perturbation_pools/
    plot_perturbation_pool_distributions.jl

  06_sampling_rules/
    select_window_samples.jl

  90_validation/
    validate_holdout_repeats.jl
    plot_holdout_validation.jl
```

Shared code lives in `lib/`. The modular driver in `workflow/` is the main
entry point. The `scripts/` folder now contains review, export, validation, and
sampling utilities that the modular workflow can call, rather than a second
independent builder.

Recommended Level 2 execution flow:

Run the modular driver:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/workflow/run_level2_workflow.jl
```

The workflow config controls which review outputs are generated:

```text
workflow/level2_workflow_config.toml
```

Use individual scripts only when rerunning one specific review task:

- `scripts/00_screen_input_data/plot_marginal_hist_screening.jl` for raw
  marginal histogram screening.
- `scripts/02_export_tables/export_level2_tables.jl` for CSV summaries.
- `scripts/03_review_joint_clusters/plot_joint_clusters.jl` for the main
  Step 2.3 joint-cluster figures.
- `scripts/03_review_joint_clusters/plot_joint_clusters_3d.jl` for
  supplementary 3D joint-cluster views.
- `scripts/03_review_joint_clusters/run_joint_cluster_sensitivity.jl` for
  minimum-cluster-fraction sensitivity checks.
- `scripts/03_review_joint_clusters/run_joint_cluster_bootstrap.jl` for
  optional cluster-stability bootstrap checks.
- `scripts/04_review_state_libraries/plot_state_library_distributions.jl` for
  low/high state-library violin figures.
- `scripts/05_review_perturbation_pools/plot_perturbation_pool_distributions.jl`
  for local/state-wide perturbation-pool figures.
- `scripts/06_sampling_rules/select_window_samples.jl` for sampling tests or
  field-scale design preparation.
- `scripts/90_validation/` for holdout-repeat validation.

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
