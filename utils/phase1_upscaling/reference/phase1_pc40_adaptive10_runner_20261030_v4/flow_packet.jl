using TOML,SHA,LinearAlgebra,DelimitedFiles
const cfg=TOML.parsefile(ARGS[1])
const SRC=cfg["science"]
include(joinpath(SRC,"vendor/pipeline/FaultCoreUpscaling.jl"))
const U=FaultCoreUpscaling
const Pc=U.Pc
const P=Pc
include(joinpath(SRC,"vendor/backend/Backend.jl"))
include(joinpath(SRC,"vendor/backend/MemoryProbe.jl"))
include(joinpath(SRC,"vendor/kernel/SelectStates.jl"))
using .SelectStates:select_states
include(joinpath(SRC,"vendor/kernel/Kernel.jl"))
include(joinpath(SRC,"vendor/PressurePlan.jl"))
include(joinpath(SRC,"ClayPc40.jl"))
include(joinpath(SRC,"AdaptiveSelector.jl"))
include(joinpath(SRC,"DrainageCache.jl"))
include(joinpath(SRC,"CachedAdaptiveCurve.jl"))
const RUN=TOML.parsefile(joinpath(SRC,"runtime.toml"))
@assert string(VERSION)=="1.12.6" && Threads.nthreads()==2
@assert bytes2hex(open(sha256,RUN["bridge"]))==RUN["bridge_sha256"]
BLAS.set_num_threads(2);U.Flow.bridge!(RUN["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,RUN["pipeline_hash"])
U.Diagnostics.reset!(RUN["pipeline_hash"])
function durable(path,value)
    if ispath(path)
        @assert TOML.parsefile(path)==value
        return
    end
    open(path,"w") do io
        TOML.print(io,value);flush(io);@assert ccall(:fsync,Cint,(Cint,),fd(io))==0
    end
end
function one(r)
    work=r["work"];raw=r["raw"];geo=r["geometry"]
    x=Pc.read_input(raw);g=U.read_geometry(geo)
    @assert x.core==r["slot"]==g.core && x.settings.co2_brine_theta_deg==40
    @assert x.settings.clay_endpoint==.8 && x.settings.sand_endpoint==.95
    master=ClayPc40.reference(SRC,r["sample"],"master")
    table=ClayPc40.clay_table(x.tables[2],master)
    @assert maximum(abs.(table.-x.tables[2]))<5e-8
    x=Pc.CoreInput(x.core,x.perm,x.phi,x.volume,x.edges,x.inlet,x.outlet,x.material,
                  (x.tables[1],table),x.pressures,x.dictionary,x.settings)
    pp,pool=dictionary(x);out=joinpath(r["rawdir"],"planned");mkpath(out)
    raw2=joinpath(out,"raw.bin");write_raw(raw2,x,pp,pool)
    h=open(sha256,raw2);g=U.Geometry(g.core,g.axis,g.length,g.area,h,g.net)
    x=Pc.read_input(raw2);@assert U.validate_geometry(x,g,h)
    model,_=Pc.prepare_model(x)
    @assert length(pp)==100 && pp[end]>=maximum(model.endpoint) && pp[2]<minimum(model.entry)
    proof=Dict("record_sha256"=>r["record_sha256"],"method"=>cfg["method"],"slot"=>x.core,
        "sample"=>r["sample"],"contact_angle_deg"=>40,"raw_sha256"=>bytes2hex(h),
        "geometry_unplanned_sha256"=>bytes2hex(open(sha256,geo)),"entry_pressure_pa"=>minimum(model.entry[model.inlet]),
        "clay_endpoint"=>.8,"sand_endpoint"=>.95,"pressure_seed_dictionary"=>"zero only; new core endpoint-derived grid",
        "maximum_local_endpoint_pa"=>maximum(model.endpoint),"maximum_pressure_pa"=>pp[end],"passed"=>true)
    durable(joinpath(work,"input/INPUT_PASS.toml"),proof)
    durable(joinpath(work,"SOLVE_STARTED.toml"),Dict("record_sha256"=>r["record_sha256"],"job"=>ENV["SLURM_JOB_ID"],"method"=>cfg["method"]))
    output=joinpath(work,"flow")
    identity=(core=r["key"],raw_sha256=proof["raw_sha256"],
              geometry_sha256=proof["geometry_unplanned_sha256"],method=cfg["method"])
    pc,kr=run_cached_adaptive(x,g,output,identity)
    s=TOML.parsefile(joinpath(output,"summary.toml"));life=s["native_lifetime"]
    @assert length(pc)==100 && 2<=length(kr)<=10 && s["same_mip_microstates"] && s["phase_solver_calls"]<=20
    @assert life["numeric_created"]==life["numeric_released"] && life["symbolic_created"]==life["symbolic_released"]
    @assert s["intrinsic_solver_calls"]==1 && all(diff([r.sg_physical for r in pc]).>=0)
    @assert s["cached_states_verified"]==100 && s["cached_queries_verified"]==length(kr)
    @assert s["replay_states_checked"]==0 && s["reference_states_checked"]==0
    @assert s["state_access"]=="verified_per_core_cache_v1" && s["cache_payload_bytes"]<=512*1024^2
    implementation=Dict(n=>bytes2hex(open(sha256,joinpath(SRC,n))) for n in
        ("DrainageCache.jl","CachedAdaptiveCurve.jl"))
    receipt=Dict("state_cache_implementation"=>implementation,"passed"=>true,"record_sha256"=>r["record_sha256"],"method"=>cfg["method"],"input"=>proof,
        "artifacts"=>Dict(n=>bytes2hex(open(sha256,joinpath(output,n))) for n in
            ("pc100.csv","kr.csv","selection.csv","summary.toml","flow_calls.jsonl")))
    durable(joinpath(output,"CORE_PASS.toml"),receipt)
    println("FLOW_PASS ",r["key"]);flush(stdout)
end
for (i,r) in enumerate(cfg["cases"])
    try
        one(r)
    catch err
        durable(joinpath(r["work"],"FLOW_ERROR.toml"),Dict("error"=>sprint(showerror,err,catch_backtrace()),"passed"=>false))
        println("FLOW_HOLD ",r["key"]," ",sprint(showerror,err));flush(stdout)
    end
    GC.gc(true);GC.gc(true)
    println("MEMORY ",MemoryProbe.after_gc());flush(stdout)
    if get(cfg,"stop_after",0)==i
        println("QUALIFICATION_CHECKPOINT_STOP");break
    end
end
