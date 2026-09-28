using TOML, SHA, LinearAlgebra
config = TOML.parsefile(ARGS[1])
include("BatchGuard.jl")
const session = BatchGuard.Session(config)
include(joinpath(config["pipeline"], "FaultCoreUpscaling.jl"))
include("Backend.jl")
include("MemoryProbe.jl")
const U = FaultCoreUpscaling
@assert string(VERSION) == "1.12.6" && Threads.nthreads() == config["threads"]
BLAS.set_num_threads(config["threads"])
@assert config["threads"] in (1, 2) && config["passes"] == length(config["cases"])
@assert all(1 <= c["slot"] <= 522 for c in config["cases"]) && 1 <= config["passes"] <= 8
U.Diagnostics.verify_receipt!(U._source_hash_impl, config["pipeline_hash"])
@assert !ispath(config["output"])
mkpath(config["output"])

function run_one_core(pass, config)
# BEGIN PINNED SCIENTIFIC BODY
    case = config["cases"][pass]
    @assert bytes2hex(open(sha256, case["raw"])) == case["raw_hash"]
    @assert bytes2hex(open(sha256, case["geometry"])) == case["geometry_hash"]
    x = U.Pc.read_input(case["raw"])
    @assert x.core == case["slot"] && length(x.pressures) == 100
    @assert x.settings.co2_brine_theta_deg == 60 && x.settings.clay_endpoint == 0.8
    @assert x.settings.sand_endpoint == 0.95
    @assert x.tables[2][end, 1] == 0.8 && x.tables[2][end, 3] == 0
    gc_before = MemoryProbe.after_gc()
    Backend.reset!()
    U.Diagnostics.reset!(config["pipeline_hash"])
    folder = joinpath(config["output"], "pass" * lpad(string(pass), 2, '0'))
    solver = Backend.Solver(U.Flow)
    println("MANAGED_MEMORY_BEGIN pass=", pass, " core=", x.core); flush(stdout)
    started = time_ns()
    try
        U.run_files(case["raw"], case["geometry"], config["bridge"], folder;
                    save_states=false, cache_states=true, phase_solver=solver)
    finally
        Backend.clear!(solver)
    end
    elapsed = (time_ns() - started)/1e9
    @assert solver.brine.factor === nothing && solver.gas.factor === nothing
    lifetime = Backend.report()
    stats = U.Diagnostics.snapshot()
    @assert get(stats["counts"], "source_receipt_hits", 0) == 2
    @assert lifetime["numeric_created"] == lifetime["numeric_released"]
    @assert lifetime["symbolic_created"] == lifetime["symbolic_released"]
    after = MemoryProbe.after_gc()
    open(joinpath(config["output"], "pass$(pass)_timing.toml"), "w") do io
        TOML.print(io, Dict("pass"=>pass, "slot"=>case["slot"], "mode"=>"managed_only", "seconds"=>elapsed,
                   "memory_before"=>gc_before, "memory_after"=>after,
                   "lifetime"=>lifetime, "diagnostics"=>stats))
    end
    println("MANAGED_MEMORY_END pass=", pass, " seconds=", elapsed); flush(stdout)
# END PINNED SCIENTIFIC BODY
    nothing
end
for pass in 1:config["passes"]
    BatchGuard.before!(session, pass)
    run_one_core(pass, config)
    BatchGuard.record!(session, pass, MemoryProbe.after_gc())
end
U.Diagnostics.verify_receipt!(U._source_hash_impl, config["pipeline_hash"])
println("MANAGED_MEMORY_MIXED_SEQUENCE_COMPLETE"); flush(stdout)

BatchGuard.finish!(session)
