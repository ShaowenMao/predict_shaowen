"""
    Level3PermeabilityCases

Build Level 3 multiple-window permeability cases from final window similarity
groups.

This step does not sample permeability values. It defines which Level 2 pool
each window should use in each multiple-window permeability case.
"""
module Level3PermeabilityCases

using Statistics

export build_multiple_window_permeability_cases,
       enumerate_binary_group_splits,
       group_distance_matrix

"""
    build_multiple_window_permeability_cases(grouping_result, distance_result; geology_id)

Build the 10-case Level 3 multiple-window permeability case set for one
geology.

The case set contains:

- 2 Independent cases,
- 4 Fault-wide low/high cases,
- 4 Grouped low/high cases.
"""
function build_multiple_window_permeability_cases(grouping_result::Dict{String, Any},
                                                 distance_result::Dict{String, Any};
                                                 geology_id::AbstractString = "g_ref")
    windows = String.(grouping_result["windows"])
    groups = Vector{Vector{String}}(grouping_result["groups"])
    normalized_distance = Matrix{Float64}(distance_result["normalized_distance"])
    non_singleton_group_ids = [idx for idx in eachindex(groups) if length(groups[idx]) > 1]
    singleton_group_ids = [idx for idx in eachindex(groups) if length(groups[idx]) == 1]

    group_split_table = enumerate_binary_group_splits(
        groups,
        windows,
        normalized_distance;
        candidate_group_ids = non_singleton_group_ids,
    )
    selected_group_split = select_group_split(group_split_table, non_singleton_group_ids, groups)
    cases = build_case_definitions(groups, selected_group_split, non_singleton_group_ids)
    case_rows = build_case_window_rows(geology_id, windows, groups, cases)

    return Dict{String, Any}(
        "geology_id" => String(geology_id),
        "windows" => windows,
        "groups" => groups,
        "non_singleton_group_ids" => non_singleton_group_ids,
        "singleton_group_ids" => singleton_group_ids,
        "group_distance_matrix" => group_distance_matrix(groups, windows, normalized_distance),
        "group_split_table" => group_split_table,
        "selected_group_split" => selected_group_split,
        "case_definitions" => cases,
        "case_window_table" => case_rows,
    )
end

"""
    enumerate_binary_group_splits(groups, windows, normalized_distance; candidate_group_ids)

Enumerate unique binary low/high group splits among non-singleton similarity
groups.

Complements are treated as the same binary split during scoring. The final
multiple-window case set later uses both orientations of the selected split.
Singleton groups are excluded because singleton windows remain independent in
Grouped low/high cases.
"""
function enumerate_binary_group_splits(groups::Vector{Vector{String}},
                                       windows::Vector{String},
                                       normalized_distance::Matrix{Float64};
                                       candidate_group_ids::Vector{Int} = collect(eachindex(groups)))
    candidate_count = length(candidate_group_ids)
    group_dist = group_distance_matrix(groups, windows, normalized_distance)
    group_sizes = length.(groups)
    rows = Vector{Dict{String, Any}}()

    candidate_count <= 1 && return rows

    # Force group 1 into side A to remove duplicate complements.
    for mask in 1:(2^candidate_count - 2)
        ((mask & 0x1) == 0x1) || continue
        side_a = [candidate_group_ids[idx] for idx in 1:candidate_count if ((mask >> (idx - 1)) & 0x1) == 0x1]
        side_b = [idx for idx in candidate_group_ids if !in(idx, side_a)]
        isempty(side_b) && continue
        push!(rows, group_split_metrics(length(rows) + 1, side_a, side_b, candidate_group_ids, group_sizes, group_dist))
    end

    sort!(rows, by = row -> (
        -Float64(row["group_split_score"]),
        Int(row["window_count_imbalance"]),
        -Float64(row["opposite_mean_distance"]),
    ))
    for (rank, row) in enumerate(rows)
        row["rank"] = rank
    end
    return rows
end

"""
    group_distance_matrix(groups, windows, normalized_distance)

Compute average normalized distances between similarity groups.
"""
function group_distance_matrix(groups::Vector{Vector{String}},
                               windows::Vector{String},
                               normalized_distance::Matrix{Float64})
    index_by_window = Dict(window => idx for (idx, window) in enumerate(windows))
    group_count = length(groups)
    group_dist = zeros(Float64, group_count, group_count)
    for a in 1:group_count
        for b in a+1:group_count
            distances = Float64[]
            for wi in groups[a], wj in groups[b]
                push!(distances, normalized_distance[index_by_window[wi], index_by_window[wj]])
            end
            value = mean(distances)
            group_dist[a, b] = value
            group_dist[b, a] = value
        end
    end
    return group_dist
end

function group_split_metrics(group_split_id::Int,
                             side_a::Vector{Int},
                             side_b::Vector{Int},
                             candidate_group_ids::Vector{Int},
                             group_sizes::Vector{Int},
                             group_dist::Matrix{Float64})
    opposite_distances = Float64[]
    same_distances = Float64[]
    for ii in 1:length(candidate_group_ids)-1
        i = candidate_group_ids[ii]
        for jj in ii+1:length(candidate_group_ids)
            j = candidate_group_ids[jj]
            same_side = (in(i, side_a) && in(j, side_a)) || (in(i, side_b) && in(j, side_b))
            if same_side
                push!(same_distances, group_dist[i, j])
            else
                push!(opposite_distances, group_dist[i, j])
            end
        end
    end

    opposite_mean = mean(opposite_distances)
    same_mean = isempty(same_distances) ? 0.0 : mean(same_distances)
    side_a_window_count = sum(group_sizes[side_a])
    side_b_window_count = sum(group_sizes[side_b])

    return Dict{String, Any}(
        "group_split_id" => group_split_id,
        "rank" => 0,
        "side_a_groups" => side_a,
        "side_b_groups" => side_b,
        "side_a_group_label" => group_label(side_a),
        "side_b_group_label" => group_label(side_b),
        "side_a_window_count" => side_a_window_count,
        "side_b_window_count" => side_b_window_count,
        "window_count_imbalance" => abs(side_a_window_count - side_b_window_count),
        "opposite_mean_distance" => opposite_mean,
        "same_mean_distance" => same_mean,
        "group_split_score" => opposite_mean - same_mean,
    )
end

function select_group_split(group_split_table::Vector{Dict{String, Any}},
                            non_singleton_group_ids::Vector{Int},
                            groups::Vector{Vector{String}})
    if isempty(non_singleton_group_ids)
        return Dict{String, Any}(
            "group_split_id" => 0,
            "rank" => 0,
            "side_a_groups" => Int[],
            "side_b_groups" => Int[],
            "side_a_group_label" => "",
            "side_b_group_label" => "",
            "side_a_window_count" => 0,
            "side_b_window_count" => 0,
            "window_count_imbalance" => 0,
            "opposite_mean_distance" => NaN,
            "same_mean_distance" => NaN,
            "group_split_score" => NaN,
            "selection_note" => "no_non_singleton_similarity_group_use_additional_independent_cases",
            "selected_pattern_label" => "additional_independent_cases",
        )
    elseif length(non_singleton_group_ids) == 1
        group_id = first(non_singleton_group_ids)
        return Dict{String, Any}(
            "group_split_id" => 0,
            "rank" => 0,
            "side_a_groups" => [group_id],
            "side_b_groups" => Int[],
            "side_a_group_label" => group_label([group_id]),
            "side_b_group_label" => "singletons independent",
            "side_a_window_count" => length(groups[group_id]),
            "side_b_window_count" => 0,
            "window_count_imbalance" => length(groups[group_id]),
            "opposite_mean_distance" => NaN,
            "same_mean_distance" => NaN,
            "group_split_score" => NaN,
            "selection_note" => "one_non_singleton_similarity_group_singletons_independent",
            "selected_pattern_label" => "$(group_label([group_id])) low/high; singletons independent",
        )
    end
    selected = deepcopy(first(group_split_table))
    selected["selection_note"] = "highest_group_split_score"
    has_singletons = any(length(group) == 1 for group in groups)
    singleton_note = has_singletons ? "; singletons independent" : ""
    selected["selected_pattern_label"] = "$(selected["side_a_group_label"]) vs $(selected["side_b_group_label"])$singleton_note"
    return selected
end

function build_case_definitions(groups::Vector{Vector{String}},
                                selected_group_split::Dict{String, Any},
                                non_singleton_group_ids::Vector{Int})
    cases = Vector{Dict{String, Any}}()
    push!(cases, independent_case(1, 1))
    push!(cases, independent_case(2, 2))

    push!(cases, fault_wide_case(3, "strong", "low"))
    push!(cases, fault_wide_case(4, "strong", "high"))
    push!(cases, fault_wide_case(5, "weak", "low"))
    push!(cases, fault_wide_case(6, "weak", "high"))

    if isempty(non_singleton_group_ids)
        push!(cases, independent_case(7, 3))
        push!(cases, independent_case(8, 4))
        push!(cases, independent_case(9, 5))
        push!(cases, independent_case(10, 6))
    elseif length(non_singleton_group_ids) == 1
        group_id = first(non_singleton_group_ids)
        push!(cases, grouped_single_group_case(7, "strong", "low", group_id))
        push!(cases, grouped_single_group_case(8, "strong", "high", group_id))
        push!(cases, grouped_single_group_case(9, "weak", "low", group_id))
        push!(cases, grouped_single_group_case(10, "weak", "high", group_id))
    else
        push!(cases, grouped_low_high_case(7, "strong", "A_low_B_high", selected_group_split))
        push!(cases, grouped_low_high_case(8, "strong", "A_high_B_low", selected_group_split))
        push!(cases, grouped_low_high_case(9, "weak", "A_low_B_high", selected_group_split))
        push!(cases, grouped_low_high_case(10, "weak", "A_high_B_low", selected_group_split))
    end

    return cases
end

function independent_case(case_id::Int, draw_id::Int)
    return Dict{String, Any}(
        "case_id" => case_id,
        "case_name" => "independent_draw_$draw_id",
        "case_category" => "Independent cases",
        "case_strength" => "independent",
        "pattern_name" => "independent",
        "orientation" => "",
        "draw_id" => draw_id,
        "group_states" => Dict{Int, String}(),
        "group_split_id" => 0,
    )
end

function fault_wide_case(case_id::Int,
                         case_strength::AbstractString,
                         state::AbstractString;
                         replicate::Bool = false)
    suffix = replicate ? "_replicate" : ""
    return Dict{String, Any}(
        "case_id" => case_id,
        "case_name" => "$(case_strength)_fault_wide_$(state)$suffix",
        "case_category" => "Fault-wide low/high cases",
        "case_strength" => String(case_strength),
        "pattern_name" => "fault_wide_$(state)",
        "orientation" => "",
        "draw_id" => replicate ? 2 : 1,
        "group_states" => Dict{Int, String}(),
        "fault_wide_state" => String(state),
        "group_split_id" => 0,
    )
end

function grouped_single_group_case(case_id::Int,
                                   case_strength::AbstractString,
                                   state::AbstractString,
                                   group_id::Int)
    return Dict{String, Any}(
        "case_id" => case_id,
        "case_name" => "$(case_strength)_grouped_g$(group_id)_$(state)_singletons_independent",
        "case_category" => "Grouped low/high cases",
        "case_strength" => String(case_strength),
        "pattern_name" => "grouped_low_high",
        "orientation" => "group_$(state)_singletons_independent",
        "draw_id" => 1,
        "group_states" => Dict(group_id => String(state)),
        "group_split_id" => 0,
    )
end

function grouped_low_high_case(case_id::Int,
                               case_strength::AbstractString,
                               orientation::AbstractString,
                               selected_group_split::Dict{String, Any})
    side_a_state = orientation == "A_low_B_high" ? "low" : "high"
    side_b_state = orientation == "A_low_B_high" ? "high" : "low"
    group_states = Dict{Int, String}()
    for group_id in Vector{Int}(selected_group_split["side_a_groups"])
        group_states[group_id] = side_a_state
    end
    for group_id in Vector{Int}(selected_group_split["side_b_groups"])
        group_states[group_id] = side_b_state
    end

    side_a_label = lowercase(replace(String(selected_group_split["side_a_group_label"]), "+" => "_"))
    side_b_label = lowercase(replace(String(selected_group_split["side_b_group_label"]), "+" => "_"))
    return Dict{String, Any}(
        "case_id" => case_id,
        "case_name" => "$(case_strength)_grouped_$(side_a_label)_$(side_a_state)_$(side_b_label)_$(side_b_state)",
        "case_category" => "Grouped low/high cases",
        "case_strength" => String(case_strength),
        "pattern_name" => "grouped_low_high",
        "orientation" => String(orientation),
        "draw_id" => 1,
        "group_states" => group_states,
        "group_split_id" => selected_group_split["group_split_id"],
    )
end

function build_case_window_rows(geology_id::AbstractString,
                                windows::Vector{String},
                                groups::Vector{Vector{String}},
                                cases::Vector{Dict{String, Any}})
    group_by_window = Dict{String, Int}()
    for (group_id, group) in enumerate(groups)
        for window in group
            group_by_window[window] = group_id
        end
    end

    rows = Vector{Dict{String, Any}}()
    for case in cases
        for window in windows
            group_id = group_by_window[window]
            state = assigned_state(case, group_id)
            pool = sampling_pool(case, state)
            push!(rows, Dict{String, Any}(
                "geology_id" => String(geology_id),
                "case_id" => case["case_id"],
                "case_name" => case["case_name"],
                "case_category" => case["case_category"],
                "case_strength" => case["case_strength"],
                "pattern_name" => case["pattern_name"],
                "orientation" => case["orientation"],
                "group_split_id" => case["group_split_id"],
                "window" => window,
                "similarity_group" => group_id,
                "assigned_state" => state,
                "sampling_mode" => state == "independent" ? "independent" : "state",
                "sampling_pool" => pool,
            ))
        end
    end
    return rows
end

function assigned_state(case::Dict{String, Any}, group_id::Int)
    if String(case["case_category"]) == "Independent cases"
        return "independent"
    elseif String(case["case_category"]) == "Fault-wide low/high cases"
        return String(case["fault_wide_state"])
    end
    group_states = Dict{Int, String}(case["group_states"])
    return haskey(group_states, group_id) ? String(group_states[group_id]) : "independent"
end

function sampling_pool(case::Dict{String, Any}, state::AbstractString)
    state == "independent" && return "full_library"
    strength = String(case["case_strength"])
    strength == "strong" && return "$(state)_local_pool"
    strength == "weak" && return "$(state)_state_wide_pool"
    error("Unknown case strength: $strength")
end

group_label(groups::Vector{Int}) = join(["G$group_id" for group_id in groups], "+")

end
