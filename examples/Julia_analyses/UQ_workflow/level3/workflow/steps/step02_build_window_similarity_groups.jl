"""
    step02_build_window_similarity_groups(level2_states, config)

Build window similarity groups for one geology.

This is the main Level 3 grouping step. It includes the internal operations
needed to support grouping:

1. compute normalized pairwise joint-permeability distances,
2. optionally run bootstrap QA for stable similar pairs,
3. convert the selected pairwise similarity rule into final window similarity
   groups using the all-pairs grouping rule.
"""
function step02_build_window_similarity_groups(level2_states::Dict{String, Dict{String, Any}},
                                               config::Dict{String, Any})
    distance_result = compute_grouping_distances(level2_states, config)
    bootstrap_result = Dict{String, Any}()
    grouping_mode = lowercase(String(config["grouping_mode"]))

    if Bool(config["run_bootstrap_grouping_qa"]) || grouping_mode == "bootstrap"
        bootstrap_result = run_grouping_bootstrap_qa(level2_states, config)
    end

    grouping_result = build_similarity_groups_from_selected_rule(
        distance_result,
        bootstrap_result,
        config,
    )

    return Dict{String, Any}(
        "distance_result" => distance_result,
        "bootstrap_result" => bootstrap_result,
        "grouping_result" => grouping_result,
    )
end

"""
    compute_grouping_distances(level2_states, config)

Compute normalized pairwise joint-permeability distances used for window
similarity grouping.
"""
function compute_grouping_distances(level2_states::Dict{String, Dict{String, Any}},
                                    config::Dict{String, Any})
    String(config["distance_metric"]) == "energy_log_unit" ||
        error("Window similarity grouping currently supports distance_metric = energy_log_unit")
    String(config["normalization"]) == "mean_internal_spread" ||
        error("Window similarity grouping currently supports normalization = mean_internal_spread")

    return Level3Distances.compute_window_distances(
        level2_states,
        String.(config["fixed_windows"]);
        sample_size = Int(config["pairwise_sample_size"]),
        random_seed = Int(config["random_seed"]),
    )
end

"""
    run_grouping_bootstrap_qa(level2_states, config)

Run the optional bootstrap QA calculation for window similarity grouping.
"""
function run_grouping_bootstrap_qa(level2_states::Dict{String, Dict{String, Any}},
                                   config::Dict{String, Any})
    return Level3Bootstrap.bootstrap_stable_pairs(
        level2_states,
        String.(config["fixed_windows"]);
        bootstrap_count = Int(config["bootstrap_count"]),
        bootstrap_sample_size = Int(config["bootstrap_sample_size"]),
        similarity_threshold = Float64(config["bootstrap_similarity_threshold"]),
        stable_pair_probability_threshold = Float64(config["stable_pair_probability_threshold"]),
        random_seed = Int(config["bootstrap_random_seed"]),
        show_progress = Bool(config["bootstrap_show_progress"]),
    )
end

"""
    build_similarity_groups_from_selected_rule(distance_result, bootstrap_result, config)

Build final window similarity groups from either full-data distances or
bootstrap stable-similar-pair probabilities.
"""
function build_similarity_groups_from_selected_rule(distance_result::Dict{String, Any},
                                                    bootstrap_result::Dict{String, Any},
                                                    config::Dict{String, Any})
    grouping_mode = lowercase(String(config["grouping_mode"]))
    if grouping_mode == "full_data"
        return Level3Grouping.build_window_similarity_groups_from_distances(
            String.(distance_result["windows"]),
            Matrix{Float64}(distance_result["normalized_distance"]);
            similarity_threshold = Float64(config["grouping_similarity_threshold"]),
        )
    elseif grouping_mode == "bootstrap"
        isempty(bootstrap_result) &&
            error("bootstrap grouping requires run_bootstrap_grouping_qa = true")
        return Level3Grouping.build_window_similarity_groups(
            String.(bootstrap_result["windows"]),
            Matrix{Float64}(bootstrap_result["stable_pair_probability"]);
            stable_pair_probability_threshold = Float64(config["stable_pair_probability_threshold"]),
        )
    else
        error("Unknown grouping_mode: $grouping_mode. Use \"full_data\" or \"bootstrap\".")
    end
end
