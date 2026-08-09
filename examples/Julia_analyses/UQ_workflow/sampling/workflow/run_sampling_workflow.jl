#!/usr/bin/env julia

using TOML
using UQWorkflow

const Sampling = UQWorkflow.IndependentFullFaultSampling
const DEFAULT_CONFIG = normpath(joinpath(@__DIR__, "independent_full_fault_v1.toml"))

function parse_cli(args::Vector{String})
    options = Dict{String, String}(
        "config" => DEFAULT_CONFIG,
        "output-root" => "",
        "phase" => "phase1",
        "only-geology" => "",
        "max-geologies" => "0",
        "deep-dive-csv" => "",
        "resume" => "true",
        "overwrite" => "false",
        "run-ensemble-qc" => "false",
        "qc-bootstrap-count" => "",
    )
    index = 1
    while index <= length(args)
        token = args[index]
        token in ("-h", "--help") && return Dict("help" => "true")
        startswith(token, "--") || error("Unexpected argument: $token")
        key = token[3:end]
        haskey(options, key) || error("Unknown option: $token")
        index == length(args) && error("Missing value after $token")
        options[key] = args[index + 1]
        index += 2
    end
    return options
end

function print_help()
    println("Generate explicit manifests for independent_full_fault_v1.")
    println()
    println("Usage:")
    println("  julia --project=examples/Julia_analyses/UQ_workflow \\")
    println("    examples/Julia_analyses/UQ_workflow/sampling/workflow/run_sampling_workflow.jl [options]")
    println()
    println("Options:")
    println("  --config <path>                 Sampling TOML configuration")
    println("  --output-root <path>            Override configured output root")
    println("  --phase <phase1|phase2>         Manifest phase to generate")
    println("  --only-geology <id[,id...]>     Restrict to explicit geology IDs")
    println("  --max-geologies <N>             Restrict ordered catalog; 0 means all")
    println("  --deep-dive-csv <path>          Phase 2 CSV containing geology_id")
    println("  --resume <true|false>            Skip valid completed geology manifests")
    println("  --overwrite <true|false>         Replace an existing incompatible manifest")
    println("  --run-ensemble-qc <true|false>   Run Phase 1 fidelity/independence QC")
    println("  --qc-bootstrap-count <N>         Override QC bootstrap count")
end

function parse_bool_cli(value::AbstractString)
    normalized = lowercase(strip(value))
    normalized in ("true", "1", "yes", "y") && return true
    normalized in ("false", "0", "no", "n") && return false
    error("Cannot parse boolean option '$value'")
end

function requested_geologies(options::Dict{String, String}, cfg::Sampling.SamplingConfig)
    maximum_count = parse(Int, options["max-geologies"])
    return Sampling.select_requested_geologies(
        cfg;
        phase = options["phase"],
        explicit_ids = options["only-geology"],
        deep_dive_csv = options["deep-dive-csv"],
        max_geologies = maximum_count,
    )
end

function completed_manifest_valid(cfg::Sampling.SamplingConfig,
                                  phase::AbstractString,
                                  geology_id::AbstractString)
    marker_path = joinpath(cfg.output_root, phase, "done", "$(geology_id).toml")
    isfile(marker_path) || return false
    marker = TOML.parsefile(marker_path)
    get(marker, "design_version", "") == cfg.design_version || return false
    get(marker, "configuration_sha256", "") == cfg.configuration_hash || return false
    get(marker, "code_commit", "") == cfg.code_commit || return false
    manifest_relative = String(get(marker, "manifest_relative_path", ""))
    isempty(manifest_relative) && return false
    manifest_path = normpath(joinpath(cfg.output_root, manifest_relative))
    isfile(manifest_path) || return false
    return Sampling.file_sha256(manifest_path) == get(marker, "manifest_sha256", "")
end

function write_run_report(cfg::Sampling.SamplingConfig,
                          phase::AbstractString,
                          geology_ids::Vector{String},
                          completed::Vector{String},
                          skipped::Vector{String},
                          combined::Dict{String, String})
    path = joinpath(cfg.output_root, phase, "$(phase)_generation_report.txt")
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "Independent full-fault sampling generation report")
        println(io, "design_version = $(cfg.design_version)")
        println(io, "phase = $phase")
        println(io, "configuration_sha256 = $(cfg.configuration_hash)")
        println(io, "code_commit = $(cfg.code_commit)")
        println(io, "code_dirty = $(cfg.code_dirty)")
        println(io, "selected_geology_count = $(length(geology_ids))")
        println(io, "generated_geology_count = $(length(completed))")
        println(io, "resumed_geology_count = $(length(skipped))")
        println(io, "combined_manifest = $(combined["path"])")
        println(io, "combined_manifest_sha256 = $(combined["sha256"])")
        println(io, "status = PASS")
    end
    return path
end

function main(args::Vector{String})
    options = parse_cli(args)
    if get(options, "help", "false") == "true"
        print_help()
        return
    end

    phase = lowercase(strip(options["phase"]))
    phase in ("phase1", "phase2") || error("--phase must be phase1 or phase2")
    options["phase"] = phase
    cfg = Sampling.read_sampling_config(options["config"];
                                        output_root_override = options["output-root"])
    geology_ids = requested_geologies(options, cfg)
    resume = parse_bool_cli(options["resume"])
    overwrite = parse_bool_cli(options["overwrite"])
    run_qc = parse_bool_cli(options["run-ensemble-qc"])
    phase == "phase2" && run_qc && error("Ensemble QC is defined for balanced Phase 1 cases")
    bootstrap_count = isempty(strip(options["qc-bootstrap-count"])) ?
        cfg.qc_bootstrap_count : parse(Int, options["qc-bootstrap-count"])
    bootstrap_count >= 0 || error("QC bootstrap count cannot be negative")

    mkpath(cfg.output_root)
    Sampling.write_effective_config(cfg)
    completed = String[]
    skipped = String[]

    println("Design: $(cfg.design_version)")
    println("Phase: $phase")
    println("Selected geologies: $(length(geology_ids))")
    println("Output root: $(cfg.output_root)")

    for (index, geology_id) in enumerate(geology_ids)
        if resume && completed_manifest_valid(cfg, phase, geology_id)
            println("[$index/$(length(geology_ids))] $geology_id: existing manifest is valid; skipping")
            push!(skipped, geology_id)
            continue
        end
        manifest_path = joinpath(cfg.output_root, phase, "manifests",
                                 "$(geology_id)_sampling_manifest.csv")
        isfile(manifest_path) && !overwrite &&
            error("Manifest already exists but is not resumable: $manifest_path")

        println("[$index/$(length(geology_ids))] $geology_id: loading and validating libraries")
        libraries = Sampling.load_geology_libraries(cfg, geology_id)
        println("[$index/$(length(geology_ids))] $geology_id: generating $phase manifest")
        rows = phase == "phase1" ?
            Sampling.generate_phase1_manifest(libraries, cfg) :
            Sampling.generate_phase2_manifest(libraries, cfg)
        outputs = Sampling.write_geology_outputs(libraries, rows, cfg, phase)
        println("[$index/$(length(geology_ids))] $geology_id: wrote $(outputs["manifest_path"])")

        if run_qc
            println("[$index/$(length(geology_ids))] $geology_id: running ensemble QC")
            qc = Sampling.run_ensemble_qc(rows, libraries, cfg;
                                          bootstrap_count = bootstrap_count)
            Sampling.write_ensemble_qc(qc, cfg, geology_id)
        end
        push!(completed, geology_id)
    end

    combined = Sampling.combine_phase_manifests(cfg, phase, geology_ids)
    report = write_run_report(cfg, phase, geology_ids, completed, skipped, combined)
    println("Combined manifest: $(combined["path"])")
    println("Combined SHA-256: $(combined["sha256"])")
    println("Report: $report")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main(ARGS)
end
