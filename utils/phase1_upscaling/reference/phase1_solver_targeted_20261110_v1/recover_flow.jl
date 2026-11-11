using TOML,SHA,LinearAlgebra,SparseArrays,Test
const cfg=TOML.parsefile(ARGS[1]);const SRC=cfg["science"]
include(joinpath(SRC,"vendor/pipeline/FaultCoreUpscaling.jl"))
const U=FaultCoreUpscaling;const Pc=U.Pc;const P=Pc
include(joinpath(SRC,"vendor/backend/Backend.jl"))
include(joinpath(SRC,"vendor/backend/MemoryProbe.jl"))
include(joinpath(SRC,"vendor/kernel/SelectStates.jl"))
using .SelectStates:select_states
include(joinpath(SRC,"vendor/kernel/Kernel.jl"))
include(joinpath(SRC,"AdaptiveSelector.jl"))
include(joinpath(SRC,"DrainageCache.jl"))
include("StrictProbe.jl");include("ResumeContract.jl")
const RUN=TOML.parsefile(joinpath(SRC,"runtime.toml"))
@assert string(VERSION)=="1.12.6" && Threads.nthreads()==2
@assert bytes2hex(open(sha256,RUN["bridge"]))==RUN["bridge_sha256"]
BLAS.set_num_threads(2);U.Flow.bridge!(RUN["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,RUN["pipeline_hash"])
U.Diagnostics.reset!(RUN["pipeline_hash"])
function durable(path,value)
    @assert !ispath(path)
    open(path,"w") do io
        TOML.print(io,value);flush(io);@assert ccall(:fsync,Cint,(Cint,),fd(io))==0
    end
end

@testset "Recovery scope and unchanged analytic gates" begin
    l=ResumeContract.Ledger([Dict("sample"=>0,"phase"=>"intrinsic","q"=>1.)])
    @test_throws AssertionError ResumeContract.take_saved!(l,1,"gas")
    @test l.next==1
    @test ResumeContract.take_saved!(l,0,"intrinsic")["q"]==1.
    @test ResumeContract.finished(l)
    @test ResumeContract.take_saved!(l,1,"gas")===nothing
    for n in (3,9),contrast in (1.,1e-6)
        e=hcat(collect(1:n-1),collect(2:n))
        net=U.Flow.Net(n,e,fill(2.,n-1),fill(2.,n-1),[1],[n],[2.],[2.])
        k=fill(contrast,n);k[1]=1.
        p,A,_,keep,_=U.Flow.prepare(net,k);signature=U.Flow.parts_hash(p,keep)
        d=StrictProbe.two_checks(U.Flow,p,A;deadline=time()+60,reference_extra=true)
        @test d["passed"]
        @test abs(d["reference_q"]-1/sum(1 ./ k))/(1/sum(1 ./ k))<=1e-9
        @test U.Flow.parts_hash(p,keep)==signature
    end
end

function recover()
    work=cfg["work"];out=joinpath(work,"flow");@assert !ispath(out);mkpath(out)
    raw=cfg["raw"];geo=cfg["geometry"]
    proof=TOML.parsefile(joinpath(work,"input/INPUT_PASS.toml"))
    @assert bytes2hex(open(sha256,raw))==proof["raw_sha256"]
    @assert bytes2hex(open(sha256,geo))==proof["geometry_unplanned_sha256"]
    input=Pc.read_input(raw);old=U.read_geometry(geo)
    g=U.Geometry(old.core,old.axis,old.length,old.area,open(sha256,raw),old.net)
    @assert U.validate_geometry(input,g,open(sha256,raw))
    @assert input.core==cfg["slot"]==proof["slot"] && proof["method"]==cfg["method"]
    @assert input.settings.co2_brine_theta_deg==40 && proof["sample"]==cfg["sample"]
    @assert input.settings.clay_endpoint==.8 && input.settings.sand_endpoint==.95
    identity=(core=cfg["key"],raw_sha256=proof["raw_sha256"],geometry_sha256=proof["geometry_unplanned_sha256"],method=cfg["method"])
    saved=TOML.parsefile(cfg["saved"]);ledger=ResumeContract.Ledger(saved["calls"])
    start=time_ns();m,audit=Pc.prepare_model(input);paths=Pc.prepare_paths(m)
    @assert length(m.pressures)==100 && m.pressures[end]>=maximum(m.endpoint)
    t=time_ns();drainage=cached_pc_pass(m,paths,identity);planned=drainage.rows
    planning_seconds=(time_ns()-t)/1e9
    Backend.reset!();solver=Backend.Solver(U.Flow)
    cache=Dict{String,Any}();results=Dict{Int,Any}()
    calls=0;visited=0;new_calls=0;reused=0;probe_reused=0;phase_seconds=0.
    try
        open(joinpath(out,"flow_calls.jsonl"),"w") do log
            function solve(kr,j,label,state_hash)
                old=ResumeContract.take_saved!(ledger,j,label)
                t=time_ns()
                if old!==nothing
                    q=old["q"];reused+=1
                    info=Dict("reused_original"=>true,"original_line_sha256"=>old["line_sha256"])
                elseif j==saved["probe_sample"] && label=="gas" && probe_reused==0
                    p,A,_,keep,_=U.Flow.prepare(g.net,kr)
                    @assert U.Flow.parts_hash(p,keep)==saved["parts_hash"]
                    @assert state_hash==saved["state_hash"]
                    q=saved["probe_q"];probe_reused+=1
                    info=Dict("reused_strict_probe"=>true,"probe_sha256"=>saved["probe_sha256"],"parts_hash"=>saved["parts_hash"])
                else
                    @assert ResumeContract.finished(ledger) && probe_reused==1 "Missing work precedes saved proof"
                    @assert label!="intrinsic" "Intrinsic recalculation forbidden"
                    if label=="gas"
                        p,A,_,keep,_=U.Flow.prepare(g.net,kr);signature=U.Flow.parts_hash(p,keep)
                        if p.n==0
                            q=0.;info=Dict("spanning_cells"=>0,"connectivity_zero"=>true)
                        else
                            d=StrictProbe.two_checks(U.Flow,p,A;deadline=time()+1800,reference_extra=true)
                            durable(joinpath(out,"gas$(j)_strict.toml"),d)
                            @assert d["passed"] "Unchanged strict correction checks failed"
                            @assert U.Flow.parts_hash(p,keep)==signature
                            q=d["reference_q"];info=Dict("strict_recovery"=>true,"parts_hash"=>signature,"metrics"=>d["metrics"],"difference"=>d["difference"])
                        end
                    else
                        q,info=solver(g.net,kr,label)
                    end
                    new_calls+=1
                end
                @assert isfinite(q) && q>=0
                dt=(time_ns()-t)/1e9
                U.json_value(log,Dict("sample"=>j,"phase"=>label,"q"=>q,"seconds"=>dt,"info"=>info,"state_hash"=>state_hash))
                println(log);flush(log);@assert ccall(:fsync,Cint,(Cint,),fd(log))==0
                q,dt
            end
            qa,_=solve(ones(m.n),0,"intrinsic","");@assert qa>0
            function query(j)
                @assert !haskey(results,j) && length(results)<10
                state,w=restore_drainage(drainage,m,j,identity);visited+=1
                key=bytes2hex(sha256(vcat(reinterpret(UInt8,w.kw),reinterpret(UInt8,w.kg))))
                if haskey(cache,key)
                    owner=cache[key];@assert U.bitsame(w.kw,owner.kw) && U.bitsame(w.kg,owner.kg)
                    qw,qg=owner.qw,owner.qg
                else
                    sh=digest_state(state,w)
                    qw,tw=solve(w.kw,j,"brine",sh);qg,tg=solve(w.kg,j,"gas",sh)
                    phase_seconds+=tw+tg;calls+=2
                    cache[key]=(;kw=copy(w.kw),kg=copy(w.kg),qw,qg)
                end
                @assert 0<=qw/qa<=1+1e-5 && 0<=qg/qa<=1+1e-5
                results[j]=(sample=j,pressure_pa=m.pressures[j],sg_physical=planned[j].sg_physical,krw=qw/qa,krg=qg/qa)
                println("RECOVERY_QUERY sample=",j," queries=",length(results));flush(stdout)
                [qw/qa,qg/qa]
            end
            chosen,order=AdaptiveSelector.select_adaptive([r.sg_physical for r in planned],query)
            @assert ResumeContract.finished(ledger) && reused==length(saved["calls"]) && probe_reused==1
            @assert calls<=20 && new_calls+reused+probe_reused==calls+1
            flowrows=[results[j] for j in chosen]
            writecsv(joinpath(out,"pc100.csv"),planned);writecsv(joinpath(out,"kr.csv"),flowrows)
            writecsv(joinpath(out,"selection.csv"),[(rank=k,sample=j,query_order=findfirst(==(j),order),actual_sg=planned[j].sg_physical,pressure_pa=planned[j].pressure_pa) for (k,j) in enumerate(chosen)])
            Backend.clear!(solver)
            summary=Dict("mode"=>"adaptive10","cells"=>m.n,"core"=>input.core,"pc_points"=>100,"kr_states"=>length(chosen),
                "selected_indices"=>chosen,"query_order"=>order,"phase_solver_calls"=>calls,"intrinsic_solver_calls"=>1,
                "state_access"=>"verified_per_core_cache_v1","cached_states_verified"=>100,"cached_queries_verified"=>visited,
                "cache_payload_bytes"=>drainage.payload_bytes,"replay_states_checked"=>0,"reference_states_checked"=>0,
                "planning_mip_seconds"=>planning_seconds,"phase_flow_seconds"=>phase_seconds,"wall_seconds"=>(time_ns()-start)/1e9,
                "intrinsic_conductance"=>qa,"intrinsic_permeability_m2"=>qa*g.length/g.area,"native_lifetime"=>Backend.report(),
                "model_audit"=>audit,"same_mip_microstates"=>true,"production_modified"=>false,
                "saved_flow_calls_reused"=>reused,"probe_calls_reused"=>probe_reused,"new_flow_calls"=>new_calls,
                "completed_flow_calls_repeated"=>0,"recovery_source"=>cfg["source"])
            durable(joinpath(out,"summary.toml"),summary)
            artifacts=Dict(n=>bytes2hex(open(sha256,joinpath(out,n))) for n in readdir(out))
            durable(joinpath(out,"CORE_PASS.toml"),Dict("passed"=>true,"method"=>cfg["method"],"record_sha256"=>cfg["record_sha256"],
                "input"=>proof,"artifacts"=>artifacts,"recovery_implementation"=>cfg["source"],
                "drainage_cache_sha256"=>bytes2hex(open(sha256,joinpath(SRC,"DrainageCache.jl")))))
        end
    finally
        Backend.clear!(solver)
    end
end
recover()
