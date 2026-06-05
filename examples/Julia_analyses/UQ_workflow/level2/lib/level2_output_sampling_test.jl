#!/usr/bin/env julia

const LEVEL2_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(LEVEL2_ROOT, "lib", "level2_io.jl"))
include(joinpath(LEVEL2_ROOT, "lib", "level2_selection.jl"))

using .Level2IO
using .Level2Selection

function parse_args(args::Vector{String})
    options = Dict(
        "config" => Level2IO.default_config_path(),
        "state-root" => normpath(joinpath(Level2IO.default_level2_output_root(), "g_ref")),
        "design" => "",
        "output-csv" => "",
        "design-out" => "",
        "random-seed" => "",
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

function print_help()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/level2/lib/level2_output_sampling_test.jl [options]")
    println()
    println("Options:")
    println("  --config <path>       Level 2 TOML config")
    println("  --state-root <path>   Root folder containing built Level 2 state MAT files")
    println("  --design <path>       Selection design CSV")
    println("                        Required columns: case_id,window,mode,state_label,perturbation_pool")
    println("                        If omitted, a built-in smoke-test design is used")
    println("  --output-csv <path>   Output CSV for selected realizations")
    println("  --design-out <path>   Output CSV for the design actually used")
    println("  --random-seed <int>   Random seed; defaults to config random_seed")
    println("  -h, --help            Show this help")
end

function main(args::Vector{String})
    opt = parse_args(args)
    config = Level2IO.read_level2_config(opt["config"])
    state_root = normpath(opt["state-root"])
    output_csv = isempty(opt["output-csv"]) ?
        joinpath(state_root, "tables", "selection_results", "selected_window_samples.csv") :
        normpath(opt["output-csv"])
    design_out = isempty(opt["design-out"]) ?
        joinpath(dirname(output_csv), "selection_design_used.csv") :
        normpath(opt["design-out"])
    random_seed = isempty(opt["random-seed"]) ? Int(config["random_seed"]) : parse(Int, opt["random-seed"])

    design_rows = isempty(opt["design"]) ?
        Level2Selection.default_selection_design(String.(config["fixed_windows"])) :
        Level2Selection.read_selection_design_csv(normpath(opt["design"]))

    states = load_states(state_root, String.(config["fixed_windows"]))
    results = Level2Selection.select_window_samples(states, design_rows; random_seed = random_seed)

    Level2IO.write_csv(design_out,
                       Level2Selection.SELECTION_DESIGN_HEADER,
                       Level2Selection.design_rows_to_csv_rows(design_rows))
    Level2IO.write_csv(output_csv,
                       Level2Selection.SELECTION_RESULT_HEADER,
                       Level2Selection.selection_results_to_csv_rows(results))

    println("Saved Level 2 selection design to $design_out")
    println("Saved Level 2 selected samples to $output_csv")
end

function load_states(state_root::AbstractString, windows::Vector{String})
    states = Dict{String, Dict{String, Any}}()
    for window in windows
        state_path = joinpath(state_root, "window_states", window, "$(window)_level2_state.mat")
        isfile(state_path) || error("Missing Level 2 state MAT file: $state_path")
        states[window] = Level2IO.load_window_state(state_path)
    end
    return states
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
