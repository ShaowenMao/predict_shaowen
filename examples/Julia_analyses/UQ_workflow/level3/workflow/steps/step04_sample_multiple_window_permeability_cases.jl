"""
    step04_sample_multiple_window_permeability_cases(level2_states, case_result, config)

Sample one actual joint PREDICT realization for every window in every Level 3
multiple-window permeability case.

This step converts the Step 3 case/window pool assignments into numeric
permeability values. Each sampled row preserves the joint triplet
`(kxx, kyy, kzz)` from one original PREDICT realization; components are never
mixed across different realizations.
"""
function step04_sample_multiple_window_permeability_cases(level2_states::Dict{String, Dict{String, Any}},
                                                          case_result::Dict{String, Any},
                                                          config::Dict{String, Any})
    case_window_rows = Vector{Dict{String, Any}}(case_result["case_window_table"])
    design_rows = level3_case_rows_to_level2_selection_design(case_window_rows)
    random_seed = Int(config["sampling_random_seed"])
    selected_rows = Level2Selection.select_window_samples(
        level2_states,
        design_rows;
        random_seed = random_seed,
    )

    sampled_rows = Dict{String, Any}[]
    for (assignment, selection) in zip(case_window_rows, selected_rows)
        push!(sampled_rows, merge_selection_with_assignment(assignment, selection))
    end

    return Dict{String, Any}(
        "geology_id" => String(config["geology_id"]),
        "random_seed" => random_seed,
        "case_count" => length(unique(string(row["case_id"]) for row in case_window_rows)),
        "sampled_rows" => sampled_rows,
    )
end

"""
    level3_case_rows_to_level2_selection_design(case_window_rows)

Convert Level 3 sampling-pool names into the compact Level 2 selector design
schema.
"""
function level3_case_rows_to_level2_selection_design(case_window_rows::Vector{Dict{String, Any}})
    design_rows = Dict{String, String}[]
    for row in case_window_rows
        push!(design_rows, Dict(
            "case_id" => string(row["case_id"]),
            "window" => string(row["window"]),
            "mode" => selection_mode_for_level2(row),
            "state_label" => state_label_for_level2(row),
            "perturbation_pool" => perturbation_pool_for_level2(row),
        ))
    end
    return design_rows
end

"""
    selection_mode_for_level2(row)

Map Level 3 assignment rows onto the Level 2 selector's `mode` field.
"""
function selection_mode_for_level2(row::Dict{String, Any})
    mode = lowercase(string(row["sampling_mode"]))
    mode == "independent" && return "independent"
    mode == "state" && return "state"
    error("Unknown Level 3 sampling_mode: $(row["sampling_mode"])")
end

"""
    state_label_for_level2(row)

Return `low` or `high` for state-conditioned rows, and blank for independent
rows.
"""
function state_label_for_level2(row::Dict{String, Any})
    mode = selection_mode_for_level2(row)
    mode == "independent" && return ""
    state = lowercase(string(row["assigned_state"]))
    state in ("low", "high") || error("State-conditioned sampling requires low/high state, got: $state")
    return state
end

"""
    perturbation_pool_for_level2(row)

Map Level 3 pool names onto the Level 2 selector's perturbation-pool aliases.
"""
function perturbation_pool_for_level2(row::Dict{String, Any})
    mode = selection_mode_for_level2(row)
    mode == "independent" && return ""
    pool = lowercase(string(row["sampling_pool"]))
    endswith(pool, "_local_pool") && return "local"
    endswith(pool, "_state_wide_pool") && return "state_wide"
    error("Unknown state-conditioned sampling_pool: $(row["sampling_pool"])")
end

"""
    merge_selection_with_assignment(assignment, selection)

Combine case/window assignment metadata with the actual sampled permeability
values returned by `Level2Selection`.
"""
function merge_selection_with_assignment(assignment::Dict{String, Any},
                                         selection::Dict{String, Any})
    window_assignment = string(assignment["window"])
    window_selection = string(selection["window"])
    window_assignment == window_selection ||
        error("Selection window mismatch: assignment=$window_assignment, selection=$window_selection")

    return Dict{String, Any}(
        "geology_id" => string(assignment["geology_id"]),
        "case_id" => Int(assignment["case_id"]),
        "case_name" => string(assignment["case_name"]),
        "case_category" => string(assignment["case_category"]),
        "case_strength" => string(assignment["case_strength"]),
        "pattern_name" => string(assignment["pattern_name"]),
        "orientation" => string(assignment["orientation"]),
        "group_split_id" => Int(assignment["group_split_id"]),
        "window" => window_assignment,
        "similarity_group" => Int(assignment["similarity_group"]),
        "assigned_state" => string(assignment["assigned_state"]),
        "sampling_mode" => string(assignment["sampling_mode"]),
        "sampling_pool" => string(assignment["sampling_pool"]),
        "selected_sample_index" => Int(selection["selected_sample_index"]),
        "source_pool" => string(selection["source_pool"]),
        "source_pool_size" => Int(selection["source_pool_size"]),
        "random_seed" => Int(selection["random_seed"]),
        "log_kxx" => Float64(selection["log_kxx"]),
        "log_kyy" => Float64(selection["log_kyy"]),
        "log_kzz" => Float64(selection["log_kzz"]),
        "perm_kxx" => Float64(selection["perm_kxx"]),
        "perm_kyy" => Float64(selection["perm_kyy"]),
        "perm_kzz" => Float64(selection["perm_kzz"]),
    )
end
