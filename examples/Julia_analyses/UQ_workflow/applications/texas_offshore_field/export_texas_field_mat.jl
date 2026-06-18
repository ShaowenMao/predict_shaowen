#!/usr/bin/env julia

using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))

"""
    export_texas_field_mat.jl

Convert Texas offshore field sampling CSV outputs to a compact MATLAB `.mat`
file for fast MRST loading.
"""

include(joinpath(@__DIR__, "lib", "texas_field_mat_export.jl"))

using .TexasFieldMatExport

function parse_args(args::Vector{String})
    opt = Dict(
        "output-root" => "D:/codex_gom/UQ_workflow/texas_offshore_field_sampling",
        "mat-path" => "",
        "validate" => "true",
    )
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg in ("-h", "--help")
            println("Usage:")
            println("  julia --project=examples/Julia_analyses/UQ_workflow examples/Julia_analyses/UQ_workflow/applications/texas_offshore_field/export_texas_field_mat.jl [--output-root <folder>] [--mat-path <file>] [--validate true|false]")
            exit(0)
        elseif startswith(arg, "--")
            key = arg[3:end]
            haskey(opt, key) || error("Unknown option $arg")
            i < length(args) || error("Missing value for $arg")
            opt[key] = args[i + 1]
            i += 2
        else
            error("Unexpected positional argument: $arg")
        end
    end
    return opt
end

function parse_bool(text::AbstractString)
    value = lowercase(strip(text))
    value in ("true", "t", "yes", "y", "1") && return true
    value in ("false", "f", "no", "n", "0") && return false
    error("Cannot parse boolean: $text")
end

function main(args::Vector{String})
    opt = parse_args(args)
    output_root = normpath(opt["output-root"])
    mat_path = TexasFieldMatExport.export_texas_field_mat(output_root; mat_path = opt["mat-path"])
    println("Wrote MAT export: $mat_path")

    if parse_bool(opt["validate"])
        result = TexasFieldMatExport.validate_mat_against_csv(output_root; mat_path = mat_path)
        println("Validation checked rows: $(result["checked_slice_window_rows"])")
        println("Validation mismatch count: $(result["mismatch_count"])")
        result["mismatch_count"] == 0 || error("MAT export does not match CSV outputs")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
