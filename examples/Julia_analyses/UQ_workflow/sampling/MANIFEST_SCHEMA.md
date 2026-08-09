# Sampling Manifest Schema v1

Each row represents one selected PREDICT realization for one geology, case,
throw window, and along-strike slice.

The required scientific identity is:

```text
geology_id + phase + case_id + window_id + slice_id
```

The replay identity is:

```text
checkpoint_hash + source_library_row + predict_realization_id + predict_seed
```

Independent rows use `source_library=full_distribution` and
`selection_method=uniform_with_replacement`. Deterministic benchmarks use
`selection_method=exact_logk_medoid` and `sampling_seed=0`.

Only rows with all of the following are part of probabilistic UQ:

```text
case_type=independent_full
use_for_probabilistic_uq=true
is_benchmark=false
is_stress_test=false
```

The manifest also stores expected joint permeability values so downstream
replay can verify that each exact PREDICT realization is reproduced before
upscaling begins.

Numeric downstream aliases are explicit and stable:

```text
independent_full: replicate_id 1-52
representative_medoid: 101
low_state_stress: 102
high_state_stress: 103
```

The canonical scientific identity remains the string `case_id`; numeric aliases
exist only for compatibility with the established replay/Pc/Kr implementation.

Phase 1 and Phase 2 are separate rectangular manifests. Phase 2 continues the
independent replicate sequence at 13 and must not be merged into the balanced
Phase 1 Level-1/Level-2 decomposition.
