"""
    run_level2_tables(workflow)

Export the standard Level 2 CSV tables from saved modular workflow outputs.

This wrapper delegates to the stable table-export script so the modular driver
and the original script path produce identical summaries.
"""
function run_level2_tables(workflow::Dict{String, Any})
    script = joinpath(@__DIR__, "..", "..", "scripts", "02_export_tables", "export_level2_tables.jl")
    run_level2_child_script(script, [
        "--state-root", workflow["output_root"],
        "--output-dir", workflow["output_root"],
    ])
end

"""
    run_level2_child_script(script, args)

Run a Level 2 Julia script in a child Julia process using the workflow project.
"""
function run_level2_child_script(script::AbstractString, args::Vector{String})
    project_root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    run(`$(Base.julia_cmd()) --project=$project_root $script $args`)
end
