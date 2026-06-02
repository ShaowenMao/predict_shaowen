#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

include(joinpath(@__DIR__, "..", "lib", "level3_io.jl"))
include(joinpath(@__DIR__, "..", "lib", "level3_distances.jl"))
include(joinpath(@__DIR__, "..", "lib", "level3_bootstrap.jl"))
include(joinpath(@__DIR__, "..", "lib", "level3_grouping.jl"))
include(joinpath(@__DIR__, "..", "lib", "level3_permeability_cases.jl"))
include(joinpath(@__DIR__, "..", "lib", "level3_plotting.jl"))

using .Level3IO
using .Level3Distances
using .Level3Bootstrap
using .Level3Grouping
using .Level3PermeabilityCases
using .Level3Plotting

include(joinpath(@__DIR__, "steps", "step01_load_level2_states.jl"))
include(joinpath(@__DIR__, "steps", "step02_build_window_similarity_groups.jl"))
include(joinpath(@__DIR__, "steps", "step03_build_multiple_window_permeability_cases.jl"))

"""
    parse_args(args)

Parse command-line arguments for the Level 3 workflow driver.
"""
function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level3IO.default_workflow_config_path(),
        "output-root" => "",
        "level2-state-root" => "",
    )

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            print_help()
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(options, key) || error("Unknown option $arg")
            i < length(args) || error("Missing value for $arg")
            options[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return options
end

"""
    print_help()

Print command-line usage for the Level 3 workflow driver.
"""
function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level3/workflow/run_level3_workflow.jl [options]")
    println()
    println("Options:")
    println("  --config <path>             Level 3 workflow TOML config")
    println("  --output-root <path>        Override output root")
    println("  --level2-state-root <path>  Override Level 2 state root")
    println("  -h, --help                  Show this help")
end

"""
    main(args)

Run the implemented Level 3 workflow steps.
"""
function main(args::Vector{String})
    opt = parse_args(args)
    config = Level3IO.read_level3_config(
        opt["config"];
        output_root_override = opt["output-root"],
        level2_state_root_override = opt["level2-state-root"],
    )
    output_root = config["output_root"]
    table_root = joinpath(output_root, "tables")
    report_root = joinpath(output_root, "reports")
    mkpath(table_root)
    mkpath(report_root)

    level2_states = Dict{String, Dict{String, Any}}()
    if Bool(config["run_load_level2_states"])
        println("Step 01: loading Level 2 window states")
        level2_states = step01_load_level2_states(config)
        write_loaded_state_summary(joinpath(table_root, "loaded_level2_states.csv"),
                                   level2_states,
                                   String.(config["fixed_windows"]))
    end

    if Bool(config["run_build_window_similarity_groups"])
        isempty(level2_states) && (level2_states = step01_load_level2_states(config))
        println("Step 02: building window similarity groups")
        step02_result = step02_build_window_similarity_groups(level2_states, config)
        distance_result = Dict{String, Any}(step02_result["distance_result"])
        bootstrap_result = Dict{String, Any}(step02_result["bootstrap_result"])
        grouping_result = Dict{String, Any}(step02_result["grouping_result"])

        save_distance_outputs(distance_result, config, table_root, report_root)
        if Bool(config["run_make_grouping_distance_figures"])
            println("Step 02: generating grouping distance figures")
            figure_paths = Level3Plotting.save_step2_diagnostic_figures(
                distance_result,
                output_root;
                threshold = Float64(config["figure_similarity_threshold"]),
                formats = String.(config["figure_formats"]),
            )
            save_figure_manifest(joinpath(table_root, "step02_grouping_distance_figure_manifest.csv"), figure_paths)
        end

        if !isempty(bootstrap_result)
            save_bootstrap_outputs(bootstrap_result, distance_result, config, table_root, report_root)
            if Bool(config["run_make_bootstrap_grouping_figures"])
                println("Step 02: generating bootstrap grouping QA figures")
                figure_paths = Level3Plotting.save_bootstrap_grouping_qa_figures(
                    bootstrap_result,
                    output_root;
                    formats = String.(config["figure_formats"]),
                )
                save_figure_manifest(joinpath(table_root, "step02_bootstrap_grouping_figure_manifest.csv"), figure_paths)
            end
        end

        save_grouping_outputs(grouping_result, bootstrap_result, distance_result, config, table_root, report_root)
        if Bool(config["run_make_similarity_group_figure"])
            println("Step 02: generating window similarity group figure")
            figure_paths = Level3Plotting.save_window_similarity_group_figures(
                grouping_result,
                output_root;
                formats = String.(config["figure_formats"]),
            )
            save_figure_manifest(joinpath(table_root, "step02_similarity_group_figure_manifest.csv"), figure_paths)
        end

        if Bool(config["run_build_multiple_window_permeability_cases"])
            println("Step 03: building multiple-window permeability cases")
            case_result = step03_build_multiple_window_permeability_cases(grouping_result, distance_result, config)
            save_multiple_window_permeability_case_outputs(case_result, config, table_root, report_root)
            if Bool(config["run_make_multiple_window_permeability_case_figure"])
                println("Step 03: generating multiple-window permeability case figure")
                figure_paths = Level3Plotting.save_multiple_window_permeability_case_figures(
                    case_result,
                    output_root;
                    formats = String.(config["figure_formats"]),
                )
                save_figure_manifest(joinpath(table_root, "step03_multiple_window_permeability_case_figure_manifest.csv"), figure_paths)
            end
        end
    end

    println("Completed Level 3 workflow at $output_root")
end

"""
    save_figure_manifest(path, figure_paths)

Write a compact manifest of generated diagnostic figures.
"""
function save_figure_manifest(path::AbstractString, figure_paths::Vector{String})
    rows = [[basename(figure_path), figure_path] for figure_path in figure_paths]
    Level3IO.write_csv(path, ["figure", "path"], rows)
end

"""
    save_bootstrap_outputs(bootstrap_result, distance_result, config, table_root, report_root)

Write Step 3 bootstrap matrices, stable-similar-pair table, and timing report.
"""
function save_bootstrap_outputs(bootstrap_result::Dict{String, Any},
                                distance_result::Dict{String, Any},
                                config::Dict{String, Any},
                                table_root::AbstractString,
                                report_root::AbstractString)
    windows = String.(bootstrap_result["windows"])
    probability = Matrix{Float64}(bootstrap_result["stable_pair_probability"])
    stable_pair_matrix = Matrix{Bool}(bootstrap_result["stable_pair_matrix"])
    mean_matrix = Matrix{Float64}(bootstrap_result["bootstrap_distance_mean"])
    median_matrix = Matrix{Float64}(bootstrap_result["bootstrap_distance_median"])
    p10_matrix = Matrix{Float64}(bootstrap_result["bootstrap_distance_p10"])
    p90_matrix = Matrix{Float64}(bootstrap_result["bootstrap_distance_p90"])

    Level3IO.write_square_matrix_csv(joinpath(table_root, "bootstrap_stable_similar_pair_probability.csv"),
                                     windows,
                                     probability)
    Level3IO.write_square_matrix_csv(joinpath(table_root, "bootstrap_normalized_distance_mean.csv"),
                                     windows,
                                     mean_matrix)
    Level3IO.write_square_matrix_csv(joinpath(table_root, "bootstrap_normalized_distance_median.csv"),
                                     windows,
                                     median_matrix)
    Level3IO.write_square_matrix_csv(joinpath(table_root, "bootstrap_normalized_distance_p10.csv"),
                                     windows,
                                     p10_matrix)
    Level3IO.write_square_matrix_csv(joinpath(table_root, "bootstrap_normalized_distance_p90.csv"),
                                     windows,
                                     p90_matrix)
    write_bool_square_matrix_csv(joinpath(table_root, "stable_similar_pair_decision_matrix.csv"),
                                 windows,
                                 stable_pair_matrix)

    full_normalized = haskey(distance_result, "normalized_distance") ?
        Matrix{Float64}(distance_result["normalized_distance"]) :
        fill(NaN, length(windows), length(windows))

    pair_rows = Vector{Vector{String}}()
    for i in 1:length(windows)-1
        for j in i+1:length(windows)
            push!(pair_rows, [
                windows[i],
                windows[j],
                string(round(full_normalized[i, j], digits = 8)),
                string(round(mean_matrix[i, j], digits = 8)),
                string(round(median_matrix[i, j], digits = 8)),
                string(round(p10_matrix[i, j], digits = 8)),
                string(round(p90_matrix[i, j], digits = 8)),
                string(round(probability[i, j], digits = 8)),
                string(stable_pair_matrix[i, j]),
            ])
        end
    end
    Level3IO.write_csv(joinpath(table_root, "stable_similar_pairs.csv"),
                       ["window_i", "window_j", "full_data_normalized_distance",
                        "bootstrap_distance_mean", "bootstrap_distance_median",
                        "bootstrap_distance_p10", "bootstrap_distance_p90",
                        "stable_similar_pair_probability", "is_stable_similar_pair"],
                       pair_rows)

    report = String[
        "Level 3 step 2 optional bootstrap grouping QA report",
        "created_at = $(Level3IO.timestamp_string())",
        "geology_id = $(config["geology_id"])",
        "bootstrap_count = $(bootstrap_result["bootstrap_count"])",
        "bootstrap_sample_size_requested = $(bootstrap_result["bootstrap_sample_size_requested"])",
        "similarity_threshold = $(bootstrap_result["similarity_threshold"])",
        "stable_pair_probability_threshold = $(bootstrap_result["stable_pair_probability_threshold"])",
        "random_seed = $(bootstrap_result["random_seed"])",
        "started_at = $(bootstrap_result["started_at"])",
        "elapsed_seconds = $(round(Float64(bootstrap_result["elapsed_seconds"]), digits = 2))",
        "windows = $(join(windows, ", "))",
        "",
        "Generated tables:",
        "  bootstrap_stable_similar_pair_probability.csv",
        "  bootstrap_normalized_distance_mean.csv",
        "  bootstrap_normalized_distance_median.csv",
        "  bootstrap_normalized_distance_p10.csv",
        "  bootstrap_normalized_distance_p90.csv",
        "  stable_similar_pair_decision_matrix.csv",
        "  stable_similar_pairs.csv",
    ]
    Level3IO.write_text_lines(joinpath(report_root, "level3_step02_bootstrap_grouping_qa_report.txt"), report)
end

"""
    save_grouping_outputs(grouping_result, bootstrap_result, distance_result, config, table_root, report_root)

Write final window similarity group tables and a compact report.
"""
function save_grouping_outputs(grouping_result::Dict{String, Any},
                               bootstrap_result::Dict{String, Any},
                               distance_result::Dict{String, Any},
                               config::Dict{String, Any},
                               table_root::AbstractString,
                               report_root::AbstractString)
    groups = Vector{Vector{String}}(grouping_result["groups"])
    group_rows = Vector{Vector{String}}()
    for (group_id, group) in enumerate(groups)
        members = join(group, "|")
        for window in group
            push!(group_rows, [
                string(group_id),
                window,
                string(length(group)),
                members,
            ])
        end
    end
    Level3IO.write_csv(joinpath(table_root, "window_similarity_groups.csv"),
                       ["group_id", "window", "group_size", "group_members"],
                       group_rows)

    Level3IO.write_csv(joinpath(table_root, "window_similarity_group_summary.csv"),
                       ["geology_id", "grouping_mode", "group_count", "similarity_group_structure",
                        "groups", "selection_rule", "valid_partition_count",
                        "within_group_pair_count", "pair_metric_label",
                        "pair_metric_direction", "within_group_pair_metric_sum",
                        "mean_within_group_pair_metric", "grouped_window_count"],
                       [[
                           string(config["geology_id"]),
                           string(grouping_result["grouping_mode"]),
                           string(grouping_result["group_count"]),
                           string(grouping_result["similarity_group_structure"]),
                           join([join(group, "+") for group in groups], ";"),
                           string(grouping_result["selection_rule"]),
                           string(grouping_result["valid_partition_count"]),
                           string(grouping_result["within_group_pair_count"]),
                           string(grouping_result["pair_metric_label"]),
                           string(grouping_result["pair_metric_direction"]),
                           string(round(Float64(grouping_result["within_group_pair_metric_sum"]), digits = 8)),
                           metric_string(grouping_result["mean_within_group_pair_metric"]),
                           string(grouping_result["grouped_window_count"]),
                       ]])
    write_grouping_pair_decisions(joinpath(table_root, "grouping_pair_decisions.csv"),
                                  grouping_result,
                                  bootstrap_result,
                                  distance_result)

    grouping_mode = string(grouping_result["grouping_mode"])
    similarity_threshold = isfinite(Float64(grouping_result["similarity_threshold"])) ?
        string(grouping_result["similarity_threshold"]) :
        string(get(bootstrap_result, "similarity_threshold", ""))
    stable_probability_threshold = isfinite(Float64(grouping_result["stable_pair_probability_threshold"])) ?
        string(grouping_result["stable_pair_probability_threshold"]) :
        "not_used"

    report = String[
        "Level 3 step 2 window similarity grouping report",
        "created_at = $(Level3IO.timestamp_string())",
        "geology_id = $(config["geology_id"])",
        "grouping_mode = $grouping_mode",
        "similarity_threshold = $similarity_threshold",
        "stable_pair_probability_threshold = $stable_probability_threshold",
        "group_count = $(grouping_result["group_count"])",
        "similarity_group_structure = $(grouping_result["similarity_group_structure"])",
        "groups = $(join([join(group, "+") for group in groups], "; "))",
        "selection_rule = $(grouping_result["selection_rule"])",
        "valid_partition_count = $(grouping_result["valid_partition_count"])",
        "within_group_pair_count = $(grouping_result["within_group_pair_count"])",
        "pair_metric_label = $(grouping_result["pair_metric_label"])",
        "pair_metric_direction = $(grouping_result["pair_metric_direction"])",
        "within_group_pair_metric_sum = $(round(Float64(grouping_result["within_group_pair_metric_sum"]), digits = 8))",
        "mean_within_group_pair_metric = $(metric_string(grouping_result["mean_within_group_pair_metric"]))",
        "grouped_window_count = $(grouping_result["grouped_window_count"])",
        "",
        "Generated tables:",
        "  window_similarity_groups.csv",
        "  window_similarity_group_summary.csv",
        "  grouping_pair_decisions.csv",
    ]
    Level3IO.write_text_lines(joinpath(report_root, "level3_step02_window_similarity_grouping_report.txt"), report)
end

"""
    save_multiple_window_permeability_case_outputs(case_result, config, table_root, report_root)

Write Step 3 binary group split, case-definition, and case-window assignment
tables.
"""
function save_multiple_window_permeability_case_outputs(case_result::Dict{String, Any},
                                                        config::Dict{String, Any},
                                                        table_root::AbstractString,
                                                        report_root::AbstractString)
    group_split_rows = Vector{Vector{String}}()
    for row in Vector{Dict{String, Any}}(case_result["group_split_table"])
        push!(group_split_rows, [
            string(row["group_split_id"]),
            string(row["rank"]),
            string(row["side_a_group_label"]),
            string(row["side_b_group_label"]),
            string(row["side_a_window_count"]),
            string(row["side_b_window_count"]),
            string(row["window_count_imbalance"]),
            metric_string(row["opposite_mean_distance"]),
            metric_string(row["same_mean_distance"]),
            metric_string(row["group_split_score"]),
        ])
    end
    Level3IO.write_csv(joinpath(table_root, "binary_group_split_candidates.csv"),
                       ["group_split_id", "rank", "side_a_groups", "side_b_groups",
                        "side_a_window_count", "side_b_window_count",
                        "window_count_imbalance", "opposite_mean_distance",
                        "same_mean_distance", "group_split_score"],
                       group_split_rows)

    selected = Dict{String, Any}(case_result["selected_group_split"])
    Level3IO.write_csv(joinpath(table_root, "selected_binary_group_split.csv"),
                       ["group_split_id", "rank", "side_a_groups", "side_b_groups",
                        "side_a_window_count", "side_b_window_count",
                        "window_count_imbalance", "opposite_mean_distance",
                        "same_mean_distance", "group_split_score", "selection_note",
                        "selected_pattern_label"],
                       [[
                           string(selected["group_split_id"]),
                           string(selected["rank"]),
                           string(selected["side_a_group_label"]),
                           string(selected["side_b_group_label"]),
                           string(selected["side_a_window_count"]),
                           string(selected["side_b_window_count"]),
                           string(selected["window_count_imbalance"]),
                           metric_string(selected["opposite_mean_distance"]),
                           metric_string(selected["same_mean_distance"]),
                           metric_string(selected["group_split_score"]),
                           string(selected["selection_note"]),
                           string(get(selected, "selected_pattern_label", "")),
                       ]])

    case_rows = Vector{Vector{String}}()
    for case in Vector{Dict{String, Any}}(case_result["case_definitions"])
        push!(case_rows, [
            string(case["case_id"]),
            string(case["case_name"]),
            string(case["case_category"]),
            string(case["case_strength"]),
            string(case["pattern_name"]),
            string(case["orientation"]),
            string(case["group_split_id"]),
            string(case["draw_id"]),
        ])
    end
    Level3IO.write_csv(joinpath(table_root, "multiple_window_permeability_case_definitions.csv"),
                       ["case_id", "case_name", "case_category", "case_strength",
                        "pattern_name", "orientation", "group_split_id", "draw_id"],
                       case_rows)

    case_window_rows = Vector{Vector{String}}()
    for row in Vector{Dict{String, Any}}(case_result["case_window_table"])
        push!(case_window_rows, [
            string(row["geology_id"]),
            string(row["case_id"]),
            string(row["case_name"]),
            string(row["case_category"]),
            string(row["case_strength"]),
            string(row["pattern_name"]),
            string(row["orientation"]),
            string(row["group_split_id"]),
            string(row["window"]),
            string(row["similarity_group"]),
            string(row["assigned_state"]),
            string(row["sampling_mode"]),
            string(row["sampling_pool"]),
        ])
    end
    Level3IO.write_csv(joinpath(table_root, "multiple_window_permeability_window_assignments.csv"),
                       ["geology_id", "case_id", "case_name", "case_category",
                        "case_strength", "pattern_name", "orientation",
                        "group_split_id", "window", "similarity_group",
                        "assigned_state", "sampling_mode", "sampling_pool"],
                       case_window_rows)

    selected_text = string(get(selected, "selected_pattern_label", ""))
    isempty(selected_text) && (selected_text = Int(selected["group_split_id"]) == 0 ?
        "none" :
        "$(selected["side_a_group_label"]) vs $(selected["side_b_group_label"])")
    report = String[
        "Level 3 step 3 multiple-window permeability case report",
        "created_at = $(Level3IO.timestamp_string())",
        "geology_id = $(config["geology_id"])",
        "selected_binary_group_split = $selected_text",
        "selected_group_split_score = $(metric_string(selected["group_split_score"]))",
        "case_count = $(length(Vector{Dict{String, Any}}(case_result["case_definitions"])))",
        "case_window_row_count = $(length(Vector{Dict{String, Any}}(case_result["case_window_table"])))",
        "",
        "Generated tables:",
        "  binary_group_split_candidates.csv",
        "  selected_binary_group_split.csv",
        "  multiple_window_permeability_case_definitions.csv",
        "  multiple_window_permeability_window_assignments.csv",
    ]
    Level3IO.write_text_lines(joinpath(report_root, "level3_step03_multiple_window_permeability_cases_report.txt"), report)
end

function metric_string(value)
    value_float = Float64(value)
    isfinite(value_float) || return string(value_float)
    return string(round(value_float, digits = 8))
end

function write_grouping_pair_decisions(path::AbstractString,
                                       grouping_result::Dict{String, Any},
                                       bootstrap_result::Dict{String, Any},
                                       distance_result::Dict{String, Any})
    windows = String.(grouping_result["windows"])
    stable_pair_matrix = grouping_result["stable_pair_matrix"]
    normalized = haskey(distance_result, "normalized_distance") ?
        Matrix{Float64}(distance_result["normalized_distance"]) :
        fill(NaN, length(windows), length(windows))
    probability = haskey(bootstrap_result, "stable_pair_probability") ?
        Matrix{Float64}(bootstrap_result["stable_pair_probability"]) :
        fill(NaN, length(windows), length(windows))

    rows = Vector{Vector{String}}()
    for i in 1:length(windows)-1
        for j in i+1:length(windows)
            push!(rows, [
                windows[i],
                windows[j],
                string(round(normalized[i, j], digits = 8)),
                string(round(probability[i, j], digits = 8)),
                string(stable_pair_matrix[i, j]),
            ])
        end
    end
    Level3IO.write_csv(path,
                       ["window_i", "window_j", "full_data_normalized_distance",
                        "stable_similar_pair_probability", "is_stable_similar_pair"],
                       rows)
end

function write_bool_square_matrix_csv(path::AbstractString,
                                      labels::Vector{String},
                                      matrix::Matrix{Bool})
    header = ["window"; labels]
    rows = Vector{Vector{String}}()
    for (i, label) in enumerate(labels)
        push!(rows, [label; [matrix[i, j] ? "true" : "false" for j in eachindex(labels)]])
    end
    Level3IO.write_csv(path, header, rows)
end

"""
    write_loaded_state_summary(path, states, windows)

Write a compact table confirming which Level 2 states were loaded.
"""
function write_loaded_state_summary(path::AbstractString,
                                    states::Dict{String, Dict{String, Any}},
                                    windows::Vector{String})
    rows = Vector{Vector{String}}()
    for window in windows
        state = states[window]
        log_perms = Matrix{Float64}(state["log_perms"])
        push!(rows, [
            window,
            string(get(state, "geology_id", "")),
            string(size(log_perms, 1)),
            string(size(log_perms, 2)),
            string(get(state, "chosen_k", "")),
            string(get(state, "best_silhouette", "")),
            string(get(state, "level2_state_path", "")),
        ])
    end
    Level3IO.write_csv(path,
                       ["window", "geology_id", "n_samples", "n_components",
                        "level2_chosen_k", "level2_best_silhouette", "level2_state_path"],
                       rows)
end

"""
    save_distance_outputs(distance_result, config, table_root, report_root)

Write Step 2 distance matrices and long-form pair table.
"""
function save_distance_outputs(distance_result::Dict{String, Any},
                               config::Dict{String, Any},
                               table_root::AbstractString,
                               report_root::AbstractString)
    windows = String.(distance_result["windows"])
    energy = Matrix{Float64}(distance_result["energy_distance"])
    normalized = Matrix{Float64}(distance_result["normalized_distance"])
    internal_spread = Vector{Float64}(distance_result["internal_spread"])

    Level3IO.write_square_matrix_csv(joinpath(table_root, "window_energy_distance_matrix.csv"),
                                     windows,
                                     energy)
    Level3IO.write_square_matrix_csv(joinpath(table_root, "window_normalized_distance_matrix.csv"),
                                     windows,
                                     normalized)

    spread_rows = [[windows[i], string(round(internal_spread[i], digits = 8))]
                   for i in eachindex(windows)]
    Level3IO.write_csv(joinpath(table_root, "window_internal_spread.csv"),
                       ["window", "mean_internal_spread"],
                       spread_rows)

    pair_rows = Vector{Vector{String}}()
    for i in 1:length(windows)-1
        for j in i+1:length(windows)
            denom = 0.5 * (internal_spread[i] + internal_spread[j])
            push!(pair_rows, [
                windows[i],
                windows[j],
                string(round(energy[i, j], digits = 8)),
                string(round(internal_spread[i], digits = 8)),
                string(round(internal_spread[j], digits = 8)),
                string(round(denom, digits = 8)),
                string(round(normalized[i, j], digits = 8)),
            ])
        end
    end
    Level3IO.write_csv(joinpath(table_root, "window_distance_pairs.csv"),
                       ["window_i", "window_j", "energy_distance",
                        "internal_spread_i", "internal_spread_j",
                        "normalization_denominator", "normalized_distance"],
                       pair_rows)

    report = String[
        "Level 3 step 2 grouping distance report",
        "created_at = $(Level3IO.timestamp_string())",
        "geology_id = $(config["geology_id"])",
        "level2_state_root = $(config["level2_state_root"])",
        "output_root = $(config["output_root"])",
        "distance_metric = $(config["distance_metric"])",
        "normalization = $(config["normalization"])",
        "pairwise_sample_size = $(config["pairwise_sample_size"])",
        "step2_figure_similarity_threshold = $(config["figure_similarity_threshold"])",
        "step2_figure_formats = $(join(String.(config["figure_formats"]), ", "))",
        "windows = $(join(windows, ", "))",
        "",
        "Generated tables:",
        "  window_energy_distance_matrix.csv",
        "  window_normalized_distance_matrix.csv",
        "  window_internal_spread.csv",
        "  window_distance_pairs.csv",
        "  step02_grouping_distance_figure_manifest.csv",
        "",
        "Generated Step 2 diagnostic figures:",
        "  figures/step02_window_distances/step02_normalized_distance_heatmap.<format>",
        "  figures/step02_window_distances/step02_sorted_window_distances.<format>",
    ]
    Level3IO.write_text_lines(joinpath(report_root, "level3_step02_grouping_distance_report.txt"), report)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
