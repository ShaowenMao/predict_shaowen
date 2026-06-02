# Level 3

Level 3 builds cross-window coupling information from the Level 2 window-state
objects.

Current implemented steps:

1. Load the six Level 2 window-state objects for one geology.
2. Build window similarity groups. This step includes:
   - computing pairwise joint-permeability distances in physical 3D
     `log10(k)` space,
   - optionally running bootstrap QA for stable similar pairs,
   - converting pairwise similarity decisions into final groups using the
     all-pairs rule,
   - saving the physical-order window similarity group strip.
3. Build the 10 multiple-window permeability cases. This step includes:
   - enumerating binary group split candidates,
   - scoring each candidate split,
   - selecting the best grouped low/high pattern,
   - assembling 2 Independent cases, 4 Fault-wide low/high cases, and 4 Grouped low/high
     cases,
   - saving a 10-case by 6-window permeability-assignment matrix.

The default Step 2 metric is empirical multivariate energy distance normalized
by the average internal PREDICT spread of the two windows:

```text
delta_ij = energy_distance(Wi, Wj) / (0.5 * (spread_i + spread_j))
```

Main entry point:

```powershell
julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level3/workflow/run_level3_workflow.jl
```

Configuration:

```text
level3/workflow/level3_workflow_config.toml
```

Because the Level 1/2 convergence study supports `Nsim = 2000` as a stable
PREDICT library size, the default Level 3 grouping mode uses the full 2000-run
empirical window libraries directly:

```text
stable similar pair if normalized_distance_ij <= tau_delta
```

The default full-data threshold is:

```text
tau_delta = 0.25
```

Bootstrap remains available as an optional QA option inside the window
similarity grouping step. In bootstrap mode, a stable similar pair is:

```text
P(delta_ij <= tau_delta) >= p_stable
```

where the default settings are:

```text
tau_delta = 0.25
p_stable = 0.80
bootstrap_count = 100
bootstrap_sample_size = 2000
```

Window similarity groups are formed conservatively: every pair inside a
non-singleton group must be a stable similar pair. Among all valid groupings,
the selected grouping is the most compact one:

1. minimize the number of groups,
2. if tied, use the stronger within-group similarity as the tie-breaker.

For the default full-data grouping mode, the tie-breaker is the mean
within-group normalized energy distance, so smaller is better. For bootstrap
grouping mode, the tie-breaker is the mean within-group stable-similar-pair
probability, so larger is better.

The multiple-window permeability cases use the final similarity groups to
define 10 permeability assignments:

```text
cases 1-2   Independent cases
cases 3-6   Fault-wide low/high cases
cases 7-10  Grouped low/high cases
```

Strong cases use Level 2 local pools. Weak cases use Level 2 state-wide pools.

For Grouped low/high cases, only non-singleton similarity groups participate
in grouped low/high assignment. Singleton windows remain independent because a
singleton has no stable similarity support with another window. If all six
windows are singletons, cases 7-10 become additional Independent cases.
