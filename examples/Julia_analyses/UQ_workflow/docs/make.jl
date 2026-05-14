using Documenter

const DOCS_WORKFLOW_ROOT = normpath(joinpath(@__DIR__, ".."))

include(joinpath(DOCS_WORKFLOW_ROOT, "level2", "workflow", "run_level2_workflow.jl"))
include(joinpath(DOCS_WORKFLOW_ROOT, "level2", "lib", "level2_plotting.jl"))
include(joinpath(DOCS_WORKFLOW_ROOT, "level2", "lib", "level2_selection.jl"))

makedocs(
    sitename = "UQ Workflow Documentation",
    format = Documenter.HTML(
        prettyurls = false,
        sidebar_sitename = true,
    ),
    modules = [
        Level2IO,
        Level2Core,
        Level2Plotting,
        Level2Selection,
    ],
    pages = [
        "Home" => "index.md",
        "Level 2 Workflow" => "level2_workflow.md",
        "Level 2 API" => "level2_api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)
