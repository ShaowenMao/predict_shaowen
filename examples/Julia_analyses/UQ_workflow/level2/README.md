# Level 2

Level 2 builds empirical within-window state objects from one 2000-run joint
permeability library per fixed window.

Current script layout:

- `scripts/build_level2_window_states.jl`
  builds one Level 2 object for each of the six windows.
- `scripts/summarize_level2_window_states.jl`
  compiles concise CSV and text summaries from the saved window-state objects.
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
  such as 5%, 10%, and 15%. These outputs are sensitivity checks; the 10%
  setting remains the baseline analysis setting.
- `scripts/plot_level2_window_grouping_3d.jl`
  creates one supplementary 3D grouping figure per window in both raw
  `log10(k)` space and local normal-score space.
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
   the Step 2.3 minimum-cluster-mass choice. Use the 5% and 15% figures as
   comparison views, not as automatic replacements for the 10% baseline.
