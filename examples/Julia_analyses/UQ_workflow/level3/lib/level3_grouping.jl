"""
    Level3Grouping

Convert pairwise similarity decisions into final Level 3 window similarity
groups.

The grouping rule is intentionally conservative and compact:

1. every pair inside a non-singleton group must be a stable similar pair,
2. among valid groupings, choose the grouping with the smallest number of
   groups,
3. if several valid groupings have the same group count, choose the one with
   stronger within-group similarity.
"""
module Level3Grouping

export build_window_similarity_groups,
       build_window_similarity_groups_from_distances,
       similarity_group_structure,
       all_set_partitions

"""
    build_window_similarity_groups(windows, stable_pair_probability; kwargs...)

Build final window similarity groups from bootstrap stable-similar-pair
probabilities.

A pair is stable if its probability is at least
`stable_pair_probability_threshold`. Candidate groupings are enumerated, only
groupings that satisfy the all-pairs rule are allowed, and ties among equally
compact groupings are broken by maximizing the mean within-group stable-pair
probability.
"""
function build_window_similarity_groups(windows::Vector{String},
                                        stable_pair_probability::Matrix{Float64};
                                        stable_pair_probability_threshold::Real = 0.80)
    n = length(windows)
    size(stable_pair_probability) == (n, n) ||
        error("stable_pair_probability must be a square matrix matching windows")

    stable_pair_matrix = falses(n, n)
    for i in 1:n
        for j in i+1:n
            is_stable = stable_pair_probability[i, j] >= stable_pair_probability_threshold
            stable_pair_matrix[i, j] = is_stable
            stable_pair_matrix[j, i] = is_stable
        end
    end

    result = build_groups_from_pair_matrix(
        windows,
        stable_pair_matrix,
        stable_pair_probability;
        grouping_mode = "bootstrap",
        pair_metric_label = "stable_similar_pair_probability",
        pair_metric_direction = :max,
        similarity_threshold = NaN,
        stable_pair_probability_threshold = Float64(stable_pair_probability_threshold),
    )
    result["stable_pair_probability_threshold"] = Float64(stable_pair_probability_threshold)
    return result
end

"""
    build_window_similarity_groups_from_distances(windows, normalized_distance; kwargs...)

Build final window similarity groups directly from the full-data normalized
distance matrix.

A pair is considered stable/similar when:

```text
normalized_distance_ij <= similarity_threshold
```
"""
function build_window_similarity_groups_from_distances(windows::Vector{String},
                                                       normalized_distance::Matrix{Float64};
                                                       similarity_threshold::Real = 0.25)
    n = length(windows)
    size(normalized_distance) == (n, n) ||
        error("normalized_distance must be a square matrix matching windows")

    stable_pair_matrix = falses(n, n)
    for i in 1:n
        for j in i+1:n
            is_stable = normalized_distance[i, j] <= similarity_threshold
            stable_pair_matrix[i, j] = is_stable
            stable_pair_matrix[j, i] = is_stable
        end
    end

    return build_groups_from_pair_matrix(
        windows,
        stable_pair_matrix,
        normalized_distance;
        grouping_mode = "full_data",
        pair_metric_label = "normalized_energy_distance",
        pair_metric_direction = :min,
        similarity_threshold = Float64(similarity_threshold),
        stable_pair_probability_threshold = NaN,
    )
end

function build_groups_from_pair_matrix(windows::Vector{String},
                                       stable_pair_matrix::AbstractMatrix{Bool},
                                       pair_metric_matrix::Matrix{Float64};
                                       grouping_mode::AbstractString,
                                       pair_metric_label::AbstractString,
                                       pair_metric_direction::Symbol,
                                       similarity_threshold::Real,
                                       stable_pair_probability_threshold::Real)
    n = length(windows)
    best_partition = Vector{Vector{Int}}()
    best_metrics = Dict{String, Any}()
    valid_partition_count = 0
    for partition in all_set_partitions(collect(1:n))
        is_valid_partition(partition, stable_pair_matrix) || continue
        valid_partition_count += 1
        metrics = partition_metrics(partition, pair_metric_matrix, pair_metric_direction)
        if isempty(best_metrics) || is_better_partition(metrics, best_metrics, pair_metric_direction)
            best_metrics = metrics
            best_partition = partition
        end
    end
    isempty(best_partition) && error("No valid window similarity grouping was found")

    ordered_groups = sort_groups(best_partition)
    group_labels = [[windows[idx] for idx in group] for group in ordered_groups]
    selection_rule = pair_metric_direction == :min ?
        "minimize_group_count_then_minimize_mean_within_group_$(pair_metric_label)" :
        "minimize_group_count_then_maximize_mean_within_group_$(pair_metric_label)"

    return Dict{String, Any}(
        "windows" => windows,
        "grouping_mode" => String(grouping_mode),
        "selection_rule" => selection_rule,
        "pair_metric_label" => String(pair_metric_label),
        "pair_metric_direction" => String(pair_metric_direction),
        "similarity_threshold" => Float64(similarity_threshold),
        "stable_pair_probability_threshold" => Float64(stable_pair_probability_threshold),
        "stable_pair_matrix" => stable_pair_matrix,
        "group_indices" => ordered_groups,
        "groups" => group_labels,
        "group_count" => length(ordered_groups),
        "similarity_group_structure" => similarity_group_structure(ordered_groups),
        "valid_partition_count" => valid_partition_count,
        "within_group_pair_count" => best_metrics["within_group_pair_count"],
        "within_group_pair_metric_sum" => best_metrics["within_group_pair_metric_sum"],
        "mean_within_group_pair_metric" => best_metrics["mean_within_group_pair_metric"],
        "grouped_window_count" => best_metrics["grouped_window_count"],
        "largest_group_size" => best_metrics["largest_group_size"],
        "captured_stable_pair_count" => best_metrics["within_group_pair_count"],
        "captured_stable_pair_score_sum" => best_metrics["within_group_pair_metric_sum"],
        "pair_score_label" => String(pair_metric_label),
    )
end

"""
    all_set_partitions(items)

Enumerate all set partitions for a small vector of item ids.
"""
function all_set_partitions(items::Vector{Int})
    isempty(items) && return [Vector{Vector{Int}}()]
    first_item = first(items)
    rest_partitions = all_set_partitions(items[2:end])
    partitions = Vector{Vector{Vector{Int}}}()

    for partition in rest_partitions
        new_partition = Vector{Vector{Int}}()
        push!(new_partition, [first_item])
        append!(new_partition, deepcopy(partition))
        push!(partitions, new_partition)
        for group_idx in eachindex(partition)
            new_partition = deepcopy(partition)
            push!(new_partition[group_idx], first_item)
            push!(partitions, new_partition)
        end
    end

    return partitions
end

function is_valid_partition(partition::Vector{Vector{Int}},
                            stable_pair_matrix::AbstractMatrix{Bool})
    for group in partition
        length(group) <= 1 && continue
        for a in 1:length(group)-1
            for b in a+1:length(group)
                stable_pair_matrix[group[a], group[b]] || return false
            end
        end
    end
    return true
end

function partition_metrics(partition::Vector{Vector{Int}},
                           pair_metric_matrix::Matrix{Float64},
                           pair_metric_direction::Symbol)
    within_group_pair_count = 0
    pair_metric_sum = 0.0
    grouped_window_count = 0
    largest_group_size = 0

    for group in partition
        largest_group_size = max(largest_group_size, length(group))
        length(group) > 1 && (grouped_window_count += length(group))
        length(group) <= 1 && continue
        for a in 1:length(group)-1
            for b in a+1:length(group)
                within_group_pair_count += 1
                pair_metric_sum += pair_metric_matrix[group[a], group[b]]
            end
        end
    end

    no_pairs_value = pair_metric_direction == :min ? Inf : -Inf
    mean_pair_metric = within_group_pair_count == 0 ?
        no_pairs_value :
        pair_metric_sum / within_group_pair_count

    return Dict{String, Any}(
        "group_count" => length(partition),
        "within_group_pair_count" => within_group_pair_count,
        "within_group_pair_metric_sum" => pair_metric_sum,
        "mean_within_group_pair_metric" => mean_pair_metric,
        "grouped_window_count" => grouped_window_count,
        "largest_group_size" => largest_group_size,
    )
end

function is_better_partition(candidate::Dict{String, Any},
                             incumbent::Dict{String, Any},
                             pair_metric_direction::Symbol)
    candidate["group_count"] != incumbent["group_count"] &&
        return candidate["group_count"] < incumbent["group_count"]

    candidate_mean = Float64(candidate["mean_within_group_pair_metric"])
    incumbent_mean = Float64(incumbent["mean_within_group_pair_metric"])
    if !isapprox(candidate_mean, incumbent_mean; atol = 1e-12, rtol = 1e-12)
        return pair_metric_direction == :min ?
            candidate_mean < incumbent_mean :
            candidate_mean > incumbent_mean
    end

    candidate["grouped_window_count"] != incumbent["grouped_window_count"] &&
        return candidate["grouped_window_count"] > incumbent["grouped_window_count"]
    candidate["within_group_pair_count"] != incumbent["within_group_pair_count"] &&
        return candidate["within_group_pair_count"] > incumbent["within_group_pair_count"]
    candidate["largest_group_size"] != incumbent["largest_group_size"] &&
        return candidate["largest_group_size"] > incumbent["largest_group_size"]

    return false
end

function sort_groups(partition::Vector{Vector{Int}})
    groups = [sort(group) for group in partition]
    return sort(groups, by = group -> (minimum(group), -length(group)))
end

"""
    similarity_group_structure(groups)

Return a compact group-size structure such as `4+2` or `3+2+1`.
"""
function similarity_group_structure(groups::Vector{Vector{Int}})
    sizes = sort(length.(groups), rev = true)
    return join(string.(sizes), "+")
end

end
