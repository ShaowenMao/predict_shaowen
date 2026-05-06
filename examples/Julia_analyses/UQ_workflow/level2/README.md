# Level 2

Level 2 builds empirical within-window state objects from one 2000-run joint
permeability library per fixed window.

Distance convention:

- Local ranks and local normal scores are still saved and used for state-score
  ordering.
- The default clustering, medoid, and perturbation-neighborhood distance is
  physical log-unit distance in `log10(k)` space, where `a_c = 1` for all
  components.
- Use `--distance-metric local_normal` only as a legacy/sensitivity comparison.
- The default minimum cluster fraction is 5%, so a 2000-run window requires at
  least 100 samples for a non-singleton regime.
- Low and high state libraries use regime-aware target-mass selection: extreme
  regimes are included first, and only the needed part of the adjacent boundary
  regime is added to reach the target `state_fraction`. The central library is
  selected by distance to the global medoid after excluding low/high samples.

Current script layout:

- `scripts/build_level2_window_states.jl`
  builds one Level 2 object for each of the six windows.
- `scripts/summarize_level2_window_states.jl`
  compiles concise CSV and text summaries from the saved window-state objects,
  including detailed CSV exports for the Step 2.7 perturbation neighborhoods.
- `scripts/validate_level2_window_states.jl`
  compares the built proxy objects against holdout `N2000_repeatXX.mat`
  libraries to assess stability.
- `scripts/plot_level2_window_diagnostics.jl`
  creates one multi-panel diagnostic figure per window from the saved
  Level 2 state objects.
- `scripts/plot_level2_joint_regimes.jl`
  creates a simpler regime-only figure for Step 2.3, focused on the joint
  clustering result without the later state-library overlays. It now saves
  one raw-space pairwise figure and one separate PCA figure per window.
- `scripts/plot_level2_joint_regime_sensitivity.jl`
  reruns the Step 2.3 regime detection for multiple minimum cluster fractions
  such as 5%, 10%, and 15%. These outputs are sensitivity checks; the 5%
  setting is the baseline analysis setting.
- `scripts/bootstrap_level2_joint_regimes.jl`
  optionally runs a Step 2.3 bootstrap stability check. This can be
  time-consuming for publication-grade settings such as 100 resamples per
  window. Each replicate resamples the 2000 realizations inside a window and
  reruns the same automatic `K`-selection rule, then summarizes how often the
  selected `K` matches the full-data result.
- `scripts/plot_level2_window_grouping_3d.jl`
  creates one supplementary 3D grouping figure per window in both raw
  `log10(k)` space and local normal-score space.
- `scripts/plot_level2_state_component_distributions.jl`
  creates violin-style state component distribution figures. It saves the
  original mean/median overlay figures and a companion medoid-overlay figure
  where the marker is an actual representative PREDICT realization for each
  state library.
- `scripts/plot_level2_neighbor_component_distributions.jl`
  creates Step 2.7 violin-style perturbation-neighborhood figures. Small and
  large neighborhoods are saved separately so the tight and loose perturbation
  distributions remain readable.
- `scripts/plot_marginal_hist_screening.jl`
  creates a pre-Step-2 marginal histogram screening figure in `log10(k)`
  space for each window and each component, using the proxy 2000-run
  libraries directly.
- `scripts/plot_level2_cross_window_summary.jl`
  creates one cross-window summary figure across all six windows.
- `scripts/plot_level2_holdout_validation.jl`
  visualizes the holdout-repeat validation results.

Shared code lives in `lib/`.

Recommended Level 2 execution flow:

1. run `plot_marginal_hist_screening.jl`
   This is the pre-Step-2.2 checkpoint. Review the marginal `log10(k)`
   histograms for all six windows before constructing local ranks and local
   normal scores.
2. run `build_level2_window_states.jl`
   This script starts the actual Step 2.2 work.
3. run `summarize_level2_window_states.jl`
4. run `validate_level2_window_states.jl`
5. run the Level 2 plotting scripts to save figures under a `figures/`
   folder
   Start with `plot_level2_joint_regimes.jl` for Step 2.3 review.
   Review the pairwise raw-space figure first, then the separate PCA figure.
   Use the older detailed diagnostic figure only as a secondary QC view.
   The 3D grouping figure is a supplementary spatial check rather than the
   primary decision plot.
6. run `plot_level2_joint_regime_sensitivity.jl` when checking robustness of
   the Step 2.3 minimum-cluster-mass choice. Use the 10% and 15% figures as
   comparison views, not as automatic replacements for the 5% baseline.
7. Optional: run `bootstrap_level2_joint_regimes.jl` only when checking
   whether the Step 2.3 regime choice is stable under resampling, for example
   before paper submission or reviewer response. The primary result is
   `P(K = original K)` for each window. This is not required for routine
   Level 2 development runs.
