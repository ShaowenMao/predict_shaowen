module Level2MarginalHistOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_marginal_histograms.jl"))
end

module Level2TablesOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_tables.jl"))
end

module Level2JointClustersOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_joint_clusters.jl"))
end

module Level2JointClusters3DOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_joint_clusters_3d.jl"))
end

module Level2ClusterSensitivityOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_cluster_sensitivity.jl"))
end

module Level2ClusterBootstrapOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_cluster_bootstrap.jl"))
end

module Level2StateLibrariesOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_state_libraries.jl"))
end

module Level2PerturbationPoolsOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_perturbation_pools.jl"))
end

module Level2SamplingOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_output_sampling_test.jl"))
end

module Level2ValidationOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_validation.jl"))
end

module Level2ValidationFiguresOutput
include(joinpath(@__DIR__, "..", "..", "lib", "level2_validation_figures.jl"))
end

"""
    step08_generate_input_screening(workflow)

Generate the pre-build marginal histogram screening outputs.
"""
function step08_generate_input_screening(workflow::Dict{String, Any})
    Level2MarginalHistOutput.main([
        "--config", workflow["base_config_path"],
        "--manifest", workflow["manifest_path"],
        "--output-dir", joinpath(workflow["output_root"], "figures", "marginal_hist_screening"),
    ])
end

"""
    step08_export_level2_tables(workflow)

Export standard Level 2 CSV summaries from saved window-state objects.
"""
function step08_export_level2_tables(workflow::Dict{String, Any})
    Level2TablesOutput.main([
        "--state-root", workflow["output_root"],
        "--output-dir", workflow["output_root"],
    ])
end

"""
    step08_generate_level2_figures(workflow)

Generate configured Level 2 output figures from saved window-state objects.
"""
function step08_generate_level2_figures(workflow::Dict{String, Any})
    run_cfg = workflow["run"]
    state_root = workflow["output_root"]

    if Bool(get(run_cfg, "plot_joint_clusters", true))
        Level2JointClustersOutput.main(["--state-root", state_root])
    end

    if Bool(get(run_cfg, "plot_joint_clusters_3d", false))
        Level2JointClusters3DOutput.main(["--state-root", state_root])
    end

    if Bool(get(run_cfg, "run_joint_cluster_sensitivity", false))
        Level2ClusterSensitivityOutput.main([
            "--config", workflow["base_config_path"],
            "--manifest", workflow["manifest_path"],
            "--output-dir", joinpath(workflow["output_root"], "figures", "joint_cluster_sensitivity"),
        ])
    end

    if Bool(get(run_cfg, "run_joint_cluster_bootstrap", false))
        Level2ClusterBootstrapOutput.main([
            "--config", workflow["base_config_path"],
            "--manifest", workflow["manifest_path"],
            "--state-root", state_root,
        ])
    end

    if Bool(get(run_cfg, "plot_state_libraries", true))
        Level2StateLibrariesOutput.main([
            "--config", workflow["base_config_path"],
            "--state-root", state_root,
        ])
    end

    if Bool(get(run_cfg, "plot_perturbation_pools", true))
        Level2PerturbationPoolsOutput.main([
            "--config", workflow["base_config_path"],
            "--state-root", state_root,
        ])
    end
end

"""
    step08_run_level2_sampling_test(workflow)

Run the default Level 2 sampling smoke test against saved window-state objects.
"""
function step08_run_level2_sampling_test(workflow::Dict{String, Any})
    Level2SamplingOutput.main([
        "--config", workflow["base_config_path"],
        "--state-root", workflow["output_root"],
    ])
end

"""
    step08_run_level2_validation(workflow)

Run optional holdout-repeat validation and generate validation figures.
"""
function step08_run_level2_validation(workflow::Dict{String, Any})
    validation_root = joinpath(workflow["output_root"], "validation")

    Level2ValidationOutput.main([
        "--config", workflow["base_config_path"],
        "--state-root", workflow["output_root"],
        "--output-dir", validation_root,
    ])

    Level2ValidationFiguresOutput.main([
        "--validation-root", validation_root,
    ])
end
