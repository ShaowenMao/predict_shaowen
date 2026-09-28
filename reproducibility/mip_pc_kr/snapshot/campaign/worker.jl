using TOML, SHA, LinearAlgebra, DelimitedFiles
const cfg=TOML.parsefile(ARGS[1])
function verify_pins()
    for (p,h) in cfg["pins"]; @assert bytes2hex(open(sha256,p))==h p; end
end
verify_pins()
@assert string(VERSION)=="1.12.6"
include(joinpath(cfg["kernel_root"],"SelectStates.jl"))
using .SelectStates: select_states
include(joinpath(cfg["pipeline"],"FaultCoreUpscaling.jl"))
include(joinpath(cfg["backend"],"Backend.jl"))
include(joinpath(cfg["backend"],"MemoryProbe.jl"))
const U=FaultCoreUpscaling
const Pc=U.Pc
BLAS.set_num_threads(2)
U.Flow.bridge!(cfg["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,cfg["pipeline_hash"])
U.Diagnostics.reset!(cfg["pipeline_hash"])
include(joinpath(cfg["kernel_root"],"Kernel.jl"))
if get(cfg,"bootstrap",false)
    println("CAMPAIGN_CACHE_LOAD_PASS");exit()
end
out=cfg["output"];@assert !ispath(out);mkpath(out)
records=[]
for (i,c) in enumerate(cfg["cases"])
    path=joinpath(out,c["key"])
    println("CORE_BEGIN ",c["key"]);flush(stdout)
    try
        input=Pc.read_input(c["raw"]);g=U.read_geometry(c["geometry"])
        U.validate_geometry(input,g,open(sha256,c["raw"]))
        @assert input.core==c["slot"] && length(input.pressures)==100
        @assert input.settings.co2_brine_theta_deg==60
        @assert input.settings.sand_endpoint==.95 && input.settings.clay_endpoint==.8
        pc,kr=run_curve(input,g,path,"sparse10")
        summary=TOML.parsefile(joinpath(path,"summary.toml"))
        life=summary["native_lifetime"]
        @assert life["symbolic_created"]==life["symbolic_released"]
        @assert life["numeric_created"]==life["numeric_released"]
        @assert summary["phase_solver_calls"]<=20 && summary["intrinsic_solver_calls"]==1
        @assert summary["same_mip_microstates"] && length(pc)==100 && length(kr)<=10
        m,_=Pc.prepare_model(input);entry=minimum(m.entry[m.inlet])
        audit=TOML.parsefile(c["scenario_audit"]);@assert entry==audit["entry_pressure_pa"]
        row=Dict("passed"=>true,"key"=>c["key"],"slot"=>c["slot"],
            "raw_sha256"=>bytes2hex(open(sha256,c["raw"])),
            "geometry_sha256"=>bytes2hex(open(sha256,c["geometry"])),
            "entry_pressure_pa"=>entry,"launch_lock_sha256"=>cfg["launch_lock_sha256"],
            "artifacts"=>Dict(n=>bytes2hex(open(sha256,joinpath(path,n))) for n in
                ("pc100.csv","kr.csv","selection.csv","summary.toml","flow_calls.jsonl")))
        @assert !ispath(joinpath(path,"CORE_PASS.toml"))
        open(joinpath(path,"CORE_PASS.toml"),"w") do io;TOML.print(io,row);end
        push!(records,Dict("key"=>c["key"],"passed"=>true))
        println("CORE_PASS ",c["key"]);flush(stdout)
    catch e
        msg=sprint(showerror,e,catch_backtrace())
        println(stderr,"CORE_HOLD ",c["key"]," ",msg);flush(stderr)
        push!(records,Dict("key"=>c["key"],"passed"=>false,"error"=>msg))
    end
    GC.gc(true);GC.gc(true)
    rss=MemoryProbe.after_gc()["VmRSS_kib"]
    println("AFTER_CORE_RSS_KIB ",rss);flush(stdout)
    @assert rss*1024<=cfg["memory_mib"]*1024^2/1.05 "Warm retained-memory hold"
end
verify_pins()
@assert !ispath(joinpath(out,"BATCH_RESULT.toml"))
open(joinpath(out,"BATCH_RESULT.toml"),"w") do io
    TOML.print(io,Dict("records"=>records,"all_passed"=>all(r["passed"] for r in records)))
end
