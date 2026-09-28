# Executed only in a separately allocated Linux compute job, never a login node.
using TOML, SHA, Pkg
spec = TOML.parsefile(ARGS[1])
@assert Sys.islinux() && string(VERSION) == "1.12.6"
@assert get(ENV, "SLURM_JOB_ID", "") == spec["job_id"]
@assert realpath(Base.active_project()) == realpath(spec["project_file"])
@assert ENV["JULIA_CPU_TARGET"] == spec["cpu_target"]
@assert !ispath(spec["receipt"])
before = Dict(p=>bytes2hex(open(sha256, p)) for p in spec["pinned_files"])
# No resolve/instantiate/update: missing dependencies must be staged explicitly.
Pkg.offline(true)
println("CACHE_PRECOMPILE_BEGIN"); flush(stdout)
started = time_ns()
Pkg.precompile(; strict=true)
for p in spec["load_only_files"]
    include(p)
end
@assert all(bytes2hex(open(sha256, p)) == h for (p, h) in before)
packages = Dict{String,Any}()
for (uuid, info) in Pkg.dependencies()
    packages[string(uuid)] = Dict("name"=>something(info.name, ""),
        "version"=>string(info.version), "source"=>something(info.source, ""),
        "tree_hash"=>string(info.tree_hash))
end
open(spec["receipt"], "w") do io
    TOML.print(io, Dict("status"=>"CACHE_BUILT_NOT_YET_STRICT_LOAD_QUALIFIED",
        "julia_version"=>string(VERSION), "machine"=>Sys.MACHINE,
        "cpu_target"=>spec["cpu_target"], "project"=>Base.active_project(),
        "depot_path"=>DEPOT_PATH, "pins"=>before, "packages"=>packages,
        "seconds"=>(time_ns()-started)/1e9,
        "campaign_approved"=>false); sorted=true)
end
println("CACHE_PRECOMPILE_END seconds=", (time_ns()-started)/1e9); flush(stdout)
