"""
    run_level2_sampling_test(workflow)

Run the default Level 2 sampling smoke test against the saved window states.
"""
function run_level2_sampling_test(workflow::Dict{String, Any})
    script = joinpath(@__DIR__, "..", "..", "scripts", "06_sampling_rules", "select_window_samples.jl")
    run_level2_child_script(script, [
        "--config", workflow["base_config_path"],
        "--state-root", workflow["output_root"],
    ])
end
