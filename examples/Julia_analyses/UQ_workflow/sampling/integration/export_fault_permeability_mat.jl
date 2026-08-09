#!/usr/bin/env julia

"""Export revised full-fault assignments to a compact MATLAB fieldPerm file."""

using UQWorkflow

function parse_arguments(arguments::Vector{String})
    values = Dict{String, String}()
    overwrite = false
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--overwrite"
            overwrite = true
            index += 1
            continue
        end
        startswith(argument, "--") || error("Unexpected argument: $argument")
        index == length(arguments) && error("Missing value for $argument")
        values[argument] = arguments[index + 1]
        index += 2
    end
    haskey(values, "--sampling-csv") || error("--sampling-csv is required")
    haskey(values, "--output") || error("--output is required")
    return values, overwrite
end

function main(arguments::Vector{String} = ARGS)
    values, overwrite = parse_arguments(arguments)
    source = normpath(values["--sampling-csv"])
    output = normpath(values["--output"])
    isfile(source) || error("Sampling CSV not found: $source")
    if isfile(output) && !overwrite
        error("Output exists; pass --overwrite: $output")
    end
    result = UQWorkflow.IndependentFullFaultSampling.export_field_permeability_mat(
        source; output_path = output)
    validation = UQWorkflow.IndependentFullFaultSampling.validate_field_permeability_mat(
        source, result)
    checked = validation["checked_assignment_rows"]
    mismatches = validation["mismatch_count"]
    mat_hash = validation["mat_sha256"]
    println("Saved fault-local permeability MAT: $result")
    println("Validated $checked assignments with $mismatches mismatches")
    println("MAT SHA-256: $mat_hash")
    return result
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
