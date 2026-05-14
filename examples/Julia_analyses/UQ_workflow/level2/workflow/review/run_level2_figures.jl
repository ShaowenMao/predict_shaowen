"""
    run_input_screening_figure(workflow)

Generate the pre-Step-2 marginal histogram screening outputs.
"""
function run_input_screening_figure(workflow::Dict{String, Any})
    script = joinpath(@__DIR__, "..", "..", "scripts", "00_screen_input_data", "plot_marginal_hist_screening.jl")
    run_level2_child_script(script, [
        "--config", workflow["base_config_path"],
        "--manifest", workflow["manifest_path"],
        "--output-dir", joinpath(workflow["output_root"], "figures", "marginal_hist_screening"),
    ])
end

"""
    run_level2_figures(workflow)

Generate enabled Level 2 review figures from saved window-state objects.

The workflow config controls which figure groups run. Figure generation is
delegated to the existing plotting scripts to keep visual outputs consistent
between the modular and legacy execution paths.
"""
function run_level2_figures(workflow::Dict{String, Any})
    run_cfg = workflow["run"]
    state_root = workflow["output_root"]

    if Bool(get(run_cfg, "plot_joint_clusters", true))
        script = joinpath(@__DIR__, "..", "..", "scripts", "03_review_joint_clusters", "plot_joint_clusters.jl")
        run_level2_child_script(script, ["--state-root", state_root])
    end

    if Bool(get(run_cfg, "plot_joint_clusters_3d", false))
        script = joinpath(@__DIR__, "..", "..", "scripts", "03_review_joint_clusters", "plot_joint_clusters_3d.jl")
        run_level2_child_script(script, ["--state-root", state_root])
    end

    if Bool(get(run_cfg, "run_joint_cluster_sensitivity", false))
        script = joinpath(@__DIR__, "..", "..", "scripts", "03_review_joint_clusters", "run_joint_cluster_sensitivity.jl")
        run_level2_child_script(script, [
            "--config", workflow["base_config_path"],
            "--manifest", workflow["manifest_path"],
            "--output-dir", joinpath(workflow["output_root"], "figures", "joint_cluster_sensitivity"),
        ])
    end

    if Bool(get(run_cfg, "run_joint_cluster_bootstrap", false))
        script = joinpath(@__DIR__, "..", "..", "scripts", "03_review_joint_clusters", "run_joint_cluster_bootstrap.jl")
        run_level2_child_script(script, [
            "--config", workflow["base_config_path"],
            "--manifest", workflow["manifest_path"],
            "--state-root", state_root,
        ])
    end

    if Bool(get(run_cfg, "plot_state_libraries", true))
        script = joinpath(@__DIR__, "..", "..", "scripts", "04_review_state_libraries", "plot_state_library_distributions.jl")
        run_level2_child_script(script, [
            "--config", workflow["base_config_path"],
            "--state-root", state_root,
        ])
    end

    if Bool(get(run_cfg, "plot_perturbation_pools", true))
        script = joinpath(@__DIR__, "..", "..", "scripts", "05_review_perturbation_pools", "plot_perturbation_pool_distributions.jl")
        run_level2_child_script(script, [
            "--config", workflow["base_config_path"],
            "--state-root", state_root,
        ])
    end
end
