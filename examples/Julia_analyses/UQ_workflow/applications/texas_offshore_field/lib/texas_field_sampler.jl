"""
    TexasFieldSampler

Sampling helpers that map deterministic Level 3 case/window assignments onto
field-slice draw groups.
"""
module TexasFieldSampler

using MAT

export group_case_rows,
       sample_case_draw_group,
       source_metadata_for_states,
       deterministic_draw_seed

"""
    group_case_rows(rows)

Group assignment rows by Level 3 case ID.
"""
function group_case_rows(rows::Vector{Dict{String, String}})
    grouped = Dict{Int, Vector{Dict{String, String}}}()
    for row in rows
        case_id = parse(Int, row["case_id"])
        if !haskey(grouped, case_id)
            grouped[case_id] = Dict{String, String}[]
        end
        push!(grouped[case_id], row)
    end
    for case_id in keys(grouped)
        sort!(grouped[case_id]; by = row -> row["window"])
    end
    return grouped
end

"""
    sample_case_draw_group(states, case_rows; random_seed)

Sample one actual joint realization per window for one case and one draw group.
"""
function sample_case_draw_group(states::Dict{String, Dict{String, Any}},
                                case_rows::Vector{Dict{String, String}};
                                random_seed::Integer,
                                source_metadata::Dict{String, Dict{String, Any}} = Dict{String, Dict{String, Any}}())
    design_rows = [case_row_to_selection_design(row) for row in case_rows]
    selected_rows = Main.Level2Selection.select_window_samples(
        states,
        design_rows;
        random_seed = random_seed,
    )
    return [merge_assignment_and_selection(assignment, selection, source_metadata)
            for (assignment, selection) in zip(case_rows, selected_rows)]
end

"""
    case_row_to_selection_design(row)

Convert one Level 3 assignment row into a Level 2 selection request.
"""
function case_row_to_selection_design(row::Dict{String, String})
    return Dict(
        "case_id" => row["case_id"],
        "window" => row["window"],
        "mode" => selection_mode(row),
        "state_label" => state_label(row),
        "perturbation_pool" => perturbation_pool(row),
    )
end

function selection_mode(row::Dict{String, String})
    mode = lowercase(row["sampling_mode"])
    mode == "independent" && return "independent"
    mode == "state" && return "state"
    error("Unknown sampling_mode: $(row["sampling_mode"])")
end

function state_label(row::Dict{String, String})
    selection_mode(row) == "independent" && return ""
    state = lowercase(row["assigned_state"])
    state in ("low", "high") || error("State-conditioned row requires low/high state, got $state")
    return state
end

function perturbation_pool(row::Dict{String, String})
    selection_mode(row) == "independent" && return ""
    pool = lowercase(row["sampling_pool"])
    endswith(pool, "_local_pool") && return "local"
    endswith(pool, "_state_wide_pool") && return "state_wide"
    error("Unknown sampling_pool: $(row["sampling_pool"])")
end

"""
    merge_assignment_and_selection(assignment, selection)

Combine Level 3 assignment metadata with sampled permeability values.
"""
function merge_assignment_and_selection(assignment::Dict{String, String},
                                        selection::Dict{String, Any},
                                        source_metadata::Dict{String, Dict{String, Any}})
    assignment["window"] == string(selection["window"]) ||
        error("Window mismatch: assignment=$(assignment["window"]), selection=$(selection["window"])")

    out = Dict{String, Any}()
    for (key, value) in assignment
        out[key] = value
    end
    for key in ["selected_sample_index", "source_pool", "source_pool_size",
                "random_seed", "log_kxx", "log_kyy", "log_kzz",
                "perm_kxx", "perm_kyy", "perm_kzz"]
        out[key] = selection[key]
    end
    window = string(selection["window"])
    if haskey(source_metadata, window)
        append_source_metadata!(out, source_metadata[window])
    end
    return out
end

"""
    source_metadata_for_states(states)

Read source PREDICT checkpoint metadata for each loaded Level 2 window state.
"""
function source_metadata_for_states(states::Dict{String, Dict{String, Any}})
    metadata = Dict{String, Dict{String, Any}}()
    for (window, state) in states
        source_path = string(get(state, "source_path", ""))
        isempty(source_path) && error("Level 2 state for $window is missing source_path")
        source = matread(source_path)
        meta = Dict{String, Any}(get(source, "meta", Dict{String, Any}()))
        seed_base = Int(round(Float64(meta["SeedBase"])))
        num_attempts = Int(round(Float64(get(meta, "NumAttempts", NaN))))
        num_rejected = Int(round(Float64(get(meta, "NumRejected", NaN))))
        metadata[window] = Dict{String, Any}(
            "source_checkpoint_file" => source_path,
            "source_seed_base" => seed_base,
            "source_num_attempts" => num_attempts,
            "source_num_rejected" => num_rejected,
        )
    end
    return metadata
end

"""
    append_source_metadata!(row, metadata)

Attach replay-oriented source metadata to one sampled window row.
"""
function append_source_metadata!(row::Dict{String, Any}, metadata::Dict{String, Any})
    selected_sample_index = Int(row["selected_sample_index"])
    seed_base = Int(metadata["source_seed_base"])
    num_rejected = Int(metadata["source_num_rejected"])

    row["source_checkpoint_file"] = metadata["source_checkpoint_file"]
    row["source_seed_base"] = seed_base
    row["source_num_attempts"] = metadata["source_num_attempts"]
    row["source_num_rejected"] = num_rejected
    if num_rejected == 0
        row["exact_replay_seed"] = seed_base + selected_sample_index - 1
        row["fine_scale_replay_status"] = "direct_exact_replay_possible_if_PREDICT_code_unchanged"
    else
        row["exact_replay_seed"] = ""
        row["fine_scale_replay_status"] = "requires_valid_attempt_replay_due_to_rejected_realizations"
    end
    return row
end

"""
    deterministic_draw_seed(base_seed, assignment_row, case_id, draw_group_index)

Build a stable seed for one geology/case/draw-group. This makes a subset run
reproduce the same draw as the corresponding row in a full run.
"""
function deterministic_draw_seed(base_seed::Integer,
                                 assignment_row::Dict{String, String},
                                 case_id::Integer,
                                 draw_group_index::Integer)
    scenario_index = parse(Int, assignment_row["scenario_index"])
    case_index = parse(Int, assignment_row["case_index"])
    seed = Int(base_seed) +
           scenario_index * 1_000_000 +
           case_index * 10_000 +
           Int(case_id) * 100 +
           Int(draw_group_index)
    return seed
end

end
