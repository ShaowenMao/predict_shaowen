using MAT
using SHA
using Test
using TOML
using UQWorkflow

const Sampling = UQWorkflow.IndependentFullFaultSampling

function write_fixture_csv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ','))
        for row in rows
            println(io, join(string.(row), ','))
        end
    end
end

function fixture_logk(window_index::Int, n::Int)
    values = Matrix{Float64}(undef, n, 3)
    for row in 1:n
        values[row, 1] = -5.8 + 0.035 * row + 0.08 * window_index
        values[row, 2] = -5.1 + 0.028 * row + 0.05 * sin(0.3 * row + window_index)
        values[row, 3] = -4.7 + 0.022 * row + 0.08 * cos(0.2 * row + window_index)
    end
    return values
end

function write_fixture_window(predict_root,
                              level2_root,
                              geology_id,
                              window,
                              window_index,
                              n,
                              target_count)
    logK = fixture_logk(window_index, n)
    K = 10.0 .^ logK
    realization_ids = collect(1001:(1000 + n))
    predict_seeds = collect((200_000 + 10_000 * window_index + 1):
                            (200_000 + 10_000 * window_index + n))
    low_rows = collect(1:target_count)
    high_rows = collect((n - target_count + 1):n)
    full_medoid = Sampling.exact_logk_medoid(logK, realization_ids)
    low_medoid = Sampling.exact_logk_medoid(logK, low_rows, realization_ids)
    high_medoid = Sampling.exact_logk_medoid(logK, high_rows, realization_ids)

    checkpoint_path = joinpath(predict_root, geology_id, window, "predict_runs.mat")
    mkpath(dirname(checkpoint_path))
    matwrite(checkpoint_path, Dict(
        "perms" => K,
        "meta" => Dict(
            "AcceptedSeeds" => reshape(Float64.(predict_seeds), :, 1),
            "AcceptedAttemptIndices" => reshape(Float64.(realization_ids), :, 1),
            "NumReturned" => Float64(n),
        ),
        "checkpointInfo" => Dict(
            "AcceptedSeedTracking" => true,
            "Window" => window,
        ),
    ))

    state_path = joinpath(level2_root, geology_id, "window_states", window,
                          "$(window)_level2_state.mat")
    mkpath(dirname(state_path))
    matwrite(state_path, Dict(
        "schema_version" => "level2_window_state_v1",
        "geology_id" => geology_id,
        "window" => window,
        "source_path" => checkpoint_path,
        "n_samples" => n,
        "raw_perms" => K,
        "log_perms" => logK,
        "joint_rank_score" => collect(1:n) ./ (n + 1),
        "cluster_assignments" => ones(Int, n),
        "chosen_k" => 1,
        "best_silhouette" => 0.0,
        "global_medoid_index" => full_medoid,
        "low_indices" => low_rows,
        "high_indices" => high_rows,
        "low_medoid_index" => low_medoid,
        "high_medoid_index" => high_medoid,
    ))
    return checkpoint_path
end

function build_fixture(root)
    geology_id = "s01_c001"
    windows = ["famp1", "famp2"]
    n = 100
    target_count = 20
    predict_root = joinpath(root, "predict")
    level2_root = joinpath(root, "level2")
    output_root = joinpath(root, "outputs")
    geology_catalog = joinpath(root, "geology_catalog.csv")
    library_catalog = joinpath(root, "library_catalog.csv")

    checkpoints = String[]
    for (window_index, window) in enumerate(windows)
        push!(checkpoints, write_fixture_window(predict_root, level2_root, geology_id,
                                                window, window_index, n, target_count))
    end
    write_fixture_csv(geology_catalog,
                      ["geology_id", "all_window_files_present"],
                      [[geology_id, true]])
    write_fixture_csv(library_catalog,
                      ["geology_id", "window", "sample_kind", "mat_path", "n_samples", "file_exists"],
                      [[geology_id, window, "predict_100", checkpoints[index], n, true]
                       for (index, window) in enumerate(windows)])

    repo_root = normpath(joinpath(@__DIR__, "../../../.."))
    config_path = joinpath(root, "fixture_config.toml")
    config = Dict{String, Any}(
        "design" => Dict("version" => Sampling.DESIGN_VERSION),
        "paths" => Dict(
            "repo_root" => repo_root,
            "geology_catalog" => geology_catalog,
            "library_catalog" => library_catalog,
            "level2_root" => level2_root,
            "predict_data_root" => predict_root,
            "predict_hash_inventory" => "",
            "output_root" => output_root,
        ),
        "geometry" => Dict("windows" => windows, "n_slices" => 5, "n_predict" => n),
        "phase1" => Dict(
            "n_independent" => 3,
            "include_representative" => true,
            "include_low_stress" => true,
            "include_high_stress" => true,
        ),
        "phase2" => Dict("n_selected_geologies" => 1, "n_additional_independent" => 4),
        "sampling" => Dict(
            "base_seed" => 1729,
            "seed_method_version" => Sampling.SEED_METHOD_VERSION,
            "with_replacement" => true,
            "cross_window" => "conditionally_independent",
            "along_strike" => "conditionally_independent",
        ),
        "medoid" => Dict(
            "metric" => "euclidean_log10k_3d",
            "recompute" => true,
            "require_stored_match" => true,
        ),
        "low_high" => Dict("target_count" => target_count),
        "provenance" => Dict(
            "predict_code_commit" => repeat("a", 40),
            "predict_method_config_hash" => repeat("b", 64),
            "require_clean_code" => false,
        ),
        "qc" => Dict("bootstrap_count" => 2, "confidence_level" => 0.95,
                     "independence_acceptance_confidence_level" => 0.99,
                     "energy_max_points" => 20),
    )
    open(config_path, "w") do io
        TOML.print(io, config; sorted = true)
    end
    cfg = Sampling.read_sampling_config(config_path)
    return cfg, geology_id
end

function write_field_perm_fixture(root)
    path = joinpath(root, "adapted_sampling.csv")
    header = [
        "adapter_schema_version", "design_version", "geology_id",
        "scenario_index", "scenario_label", "scenario_name", "case_index",
        "case_label", "faulting_depth_m", "sand_vcl", "clay_vcl", "case_id",
        "sampling_case_id", "phase", "case_type", "replicate_id", "case_name",
        "case_category", "slice_index", "window", "assigned_state",
        "sampling_mode", "sampling_pool", "selected_sample_index",
        "predict_realization_id", "draw_seed", "source_checkpoint_file",
        "checkpoint_relative_path", "checkpoint_sha256", "exact_replay_seed",
        "log_kxx", "log_kyy", "log_kzz", "perm_kxx", "perm_kyy",
        "perm_kzz", "sampling_code_commit", "sampling_method_config_hash",
        "predict_code_commit", "predict_method_config_hash",
        "use_for_probabilistic_uq", "is_benchmark", "is_stress_test",
    ]
    rows = Vector{Vector{Any}}()
    cases = [
        (1, "s01_c001_IND_001", "independent_full", 1,
         "independent_full_001", "Probabilistic UQ", "independent",
         "independent", "full_distribution", true, false, false),
        (101, "s01_c001_REP_MEDOID", "representative_medoid", 0,
         "representative_medoid", "Deterministic benchmark", "representative",
         "deterministic_benchmark", "full_distribution", false, true, false),
    ]
    for (case_id, sampling_case_id, case_type, replicate_id, case_name,
         category, assigned_state, sampling_mode, sampling_pool,
         use_uq, is_benchmark, is_stress) in cases
        for window_number in 1:6, slice_id in 1:87
            sample_id = case_type == "independent_full" ?
                mod(137 * window_number + 17 * slice_id, 2000) + 1 :
                900 + window_number
            log_values = [-5.0 + sample_id / 5000,
                          -4.5 + sample_id / 5000,
                          -4.0 + sample_id / 5000]
            sampling_seed = case_type == "independent_full" ?
                1000 + case_id + window_number : 0
            push!(rows, Any[
                "independent_full_fault_upscaling_adapter_v1",
                Sampling.DESIGN_VERSION, "s01_c001", 1, "scenario_01",
                "fixture geology", 1, "case_001", 50.0, 0.1, 0.4, case_id,
                sampling_case_id, "phase1", case_type, replicate_id, case_name,
                category, slice_id, "famp$(window_number)", assigned_state,
                sampling_mode, sampling_pool, sample_id, sample_id,
                sampling_seed,
                "predict/famp$(window_number)/predict_runs.mat",
                "data/famp$(window_number)/predict_runs.mat",
                repeat(string(window_number; base = 16), 64),
                300_000_000 + window_number * 10_000 + sample_id,
                log_values[1], log_values[2], log_values[3],
                10.0^log_values[1], 10.0^log_values[2], 10.0^log_values[3],
                repeat("a", 40), repeat("b", 64), repeat("c", 40),
                repeat("d", 64), use_uq, is_benchmark, is_stress,
            ])
        end
    end
    Sampling.write_csv(path, header, rows)
    adapter_hash = Sampling.file_sha256(path)
    write(path * ".metadata.json", """
    {
      "canonical_manifest_sha256": "$(repeat("e", 64))",
      "geology_catalog_sha256": "$(repeat("f", 64))",
      "output_sha256": "$adapter_hash"
    }
    """)
    return path
end

@testset "Stable seed and counter sampling" begin
    seed = Sampling.stable_case_seed(1729, "s01_c001", "phase1",
                                     "independent_full", 1, "famp1")
    @test seed == UInt64(1973095914248810655)
    @test [Sampling.stable_uniform_index(UInt64(123), index, 2000) for index in 1:5] ==
          [338, 1442, 369, 805, 1104]
    draws = [Sampling.stable_uniform_index(UInt64(321), index, 2) for index in 1:5]
    @test length(unique(draws)) < length(draws)
    @test all(1 .<= draws .<= 2)
end

@testset "Production seed namespace" begin
    geology_ids = ["s$(lpad(scenario, 2, '0'))_c$(lpad(case_index, 3, '0'))"
                   for scenario in 1:6 for case_index in 1:27]
    windows = ["famp$index" for index in 1:6]
    seeds = UInt64[]
    for geology_id in geology_ids, phase in ("phase1", "phase2")
        replicate_ids = phase == "phase1" ? (1:12) : (13:52)
        for replicate_id in replicate_ids, window in windows
            push!(seeds, Sampling.stable_case_seed(
                1729, geology_id, phase, "independent_full", replicate_id, window))
        end
    end
    @test length(seeds) == 50_544
    @test length(unique(seeds)) == length(seeds)
    @test all(0 .< seeds .<= UInt64(typemax(Int64)))
end

@testset "Exact logK medoid tie-break" begin
    logK = [0.0 0.0 0.0;
            0.0 0.0 0.0;
            1.0 0.0 0.0]
    realization_ids = [9, 2, 3]
    @test Sampling.exact_logk_medoid(logK, realization_ids) == 2
    @test Sampling.exact_logk_medoid(logK, [1, 3], realization_ids) == 3
end

@testset "Reduced end-to-end manifest workflow" begin
    mktempdir() do root
        cfg, geology_id = build_fixture(root)
        @test !cfg.require_clean_code
        @test !cfg.optional_arrangement_test_enabled
        @test cfg.seed_method_version == Sampling.SEED_METHOD_VERSION
        @test cfg.state_target_count / cfg.expected_n_predict == 0.20

        effective = Sampling.effective_config_dict(cfg)
        @test effective["provenance"]["require_clean_code"] == false
        @test effective["low_high"]["target_count"] == 20
        @test effective["low_high"]["target_fraction"] == 0.20
        @test effective["qc"]["independence_acceptance_confidence_level"] == 0.99
        @test effective["optional_arrangement_test"]["enabled"] == false
        effective_path = Sampling.write_effective_config(cfg)
        @test TOML.parsefile(effective_path) == effective

        raw_config = TOML.parsefile(cfg.config_path)
        invalid_arrangement = deepcopy(raw_config)
        invalid_arrangement["optional_arrangement_test"] = Dict("enabled" => true)
        invalid_arrangement_path = joinpath(root, "invalid_arrangement.toml")
        open(invalid_arrangement_path, "w") do io
            TOML.print(io, invalid_arrangement; sorted = true)
        end
        @test_throws ErrorException Sampling.read_sampling_config(invalid_arrangement_path)

        invalid_target = deepcopy(raw_config)
        invalid_target["low_high"]["target_count"] = 19
        invalid_target_path = joinpath(root, "invalid_target.toml")
        open(invalid_target_path, "w") do io
            TOML.print(io, invalid_target; sorted = true)
        end
        @test_throws ErrorException Sampling.read_sampling_config(invalid_target_path)

        invalid_seed_method = deepcopy(raw_config)
        invalid_seed_method["sampling"]["seed_method_version"] = "unsupported"
        invalid_seed_path = joinpath(root, "invalid_seed_method.toml")
        open(invalid_seed_path, "w") do io
            TOML.print(io, invalid_seed_method; sorted = true)
        end
        @test_throws ErrorException Sampling.read_sampling_config(invalid_seed_path)

        @test Sampling.load_geology_ids(cfg) == [geology_id]
        @test Sampling.select_requested_geologies(cfg; phase = "phase1") == [geology_id]
        @test Sampling.select_requested_geologies(
            cfg; phase = "phase2", explicit_ids = geology_id) == [geology_id]
        deep_dive_path = joinpath(root, "deep_dive.csv")
        write_fixture_csv(deep_dive_path, ["geology_id"], [[geology_id]])
        @test Sampling.select_requested_geologies(
            cfg; phase = "phase2", deep_dive_csv = deep_dive_path) == [geology_id]
        @test_throws ErrorException Sampling.select_requested_geologies(cfg; phase = "phase2")
        @test_throws ErrorException Sampling.select_requested_geologies(
            cfg; phase = "phase2", explicit_ids = "$geology_id,$geology_id")
        @test_throws ErrorException Sampling.select_requested_geologies(
            cfg; phase = "phase2", explicit_ids = "unknown")
        @test_throws ErrorException Sampling.select_requested_geologies(
            cfg; phase = "phase2", explicit_ids = geology_id,
            deep_dive_csv = deep_dive_path)
        @test_throws ErrorException Sampling.select_requested_geologies(
            cfg; phase = "phase1", deep_dive_csv = deep_dive_path)
        @test_throws ErrorException Sampling.select_requested_geologies(
            cfg; phase = "phase2", explicit_ids = geology_id, max_geologies = -1)

        libraries = Sampling.load_geology_libraries(cfg, geology_id)
        @test collect(keys(libraries)) |> Set == Set(cfg.windows)

        phase1 = Sampling.generate_phase1_manifest(libraries, cfg)
        @test length(phase1) == 6 * 2 * 5
        @test length(unique(row.case_id for row in phase1)) == 6
        @test Sampling.validate_phase1_manifest(phase1, libraries, cfg)

        phase1_again = Sampling.generate_phase1_manifest(libraries, cfg)
        @test Sampling.canonical_manifest_bytes(phase1, cfg) ==
              Sampling.canonical_manifest_bytes(phase1_again, cfg)

        specs = reverse(Sampling.phase1_case_specs(cfg, geology_id))
        reordered = reduce(vcat,
                           [Sampling.generate_case_manifest(libraries, cfg, spec) for spec in specs];
                           init = Sampling.ManifestRow[])
        @test Sampling.canonical_manifest_bytes(phase1, cfg) ==
              Sampling.canonical_manifest_bytes(reordered, cfg)

        independent = [row for row in phase1 if row.case_type == "independent_full"]
        @test all(row.source_library == "full_distribution" for row in independent)
        replicate_seeds = unique((row.replicate_id, row.window_id, row.sampling_seed)
                                 for row in independent if row.slice_id == 1)
        @test length(replicate_seeds) == 3 * 2
        @test length(unique(last(item) for item in replicate_seeds)) == 3 * 2

        for case_type in ("representative_medoid", "low_state_stress", "high_state_stress")
            for window in cfg.windows
                selected = [row.source_library_row for row in phase1
                            if row.case_type == case_type && row.window_id == window]
                @test length(unique(selected)) == 1
            end
        end

        phase2 = Sampling.generate_phase2_manifest(libraries, cfg)
        @test length(phase2) == 4 * 2 * 5
        @test sort(unique(row.replicate_id for row in phase2)) == collect(4:7)
        @test Sampling.validate_phase2_manifest(phase2, libraries, cfg)

        output_paths = Sampling.write_geology_outputs(libraries, phase1, cfg, "phase1")
        @test isfile(output_paths["manifest_path"])
        @test isfile(output_paths["done_path"])
        loaded = Sampling.read_manifest_csv(output_paths["manifest_path"])
        @test Sampling.canonical_manifest_bytes(loaded, cfg) ==
              Sampling.canonical_manifest_bytes(phase1, cfg)

        combined = Sampling.combine_phase_manifests(cfg, "phase1", [geology_id])
        @test isfile(combined["path"])
        @test combined["sha256"] == Sampling.file_sha256(combined["path"])

        qc = Sampling.run_ensemble_qc(phase1, libraries, cfg; bootstrap_count = 2)
        @test length(qc["distance_rows"]) == 2 * 4
        @test length(qc["correlation_rows"]) == 1
        @test qc["individual_marginal_metric_count"] == 2 * 4
        @test isfinite(qc["marginal_global_max_normalized_distance"])
        @test qc["independence_seed_structure_pass"]
        @test qc["independence_seed_stream_count"] == 3 * 2
        @test 0 < qc["independence_empirical_p_value"] <= 1
        @test qc["overall_implementation_pass"] isa Bool

        library = libraries["famp1"]
        original_id = library.realization_id[2]
        library.realization_id[2] = library.realization_id[1]
        @test_throws ErrorException Sampling.validate_library(library, cfg)
        library.realization_id[2] = original_id
        @test Sampling.validate_library(library, cfg)
    end
end

@testset "Fault permeability MAT export contract" begin
    mktempdir() do root
        source = write_field_perm_fixture(root)
        output = joinpath(root, "fault_permeability.mat")
        @test Sampling.export_field_permeability_mat(source; output_path = output) == output
        validation = Sampling.validate_field_permeability_mat(source, output)
        @test validation["checked_assignment_rows"] == 2 * 6 * 87
        @test validation["mismatch_count"] == 0

        field_perm = matread(output)["fieldPerm"]
        @test String(field_perm["schema_version"]) ==
              "independent_full_fault_field_permeability_v1"
        @test Int.(vec(field_perm["level3_case_id"])) == [1, 101]
        @test String.(vec(field_perm["case_type"])) ==
              ["independent_full", "representative_medoid"]
        @test Bool.(vec(field_perm["use_for_probabilistic_uq"])) == [true, false]
        contract = field_perm["coordinate_contract"]
        @test Bool(contract["rotation_required"])
        @test !Bool(contract["pre_rotated"])
        @test String(contract["downstream_transform_contract"]) ==
              "fault_local_to_reservoir_grid_signed_yz_v1"
    end
end
