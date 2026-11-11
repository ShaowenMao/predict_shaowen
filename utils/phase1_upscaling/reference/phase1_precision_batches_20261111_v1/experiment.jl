using TOML, SHA, Serialization, LinearAlgebra, SparseArrays, Test
const cfg = TOML.parsefile(ARGS[1])
include(joinpath(cfg["science"],"vendor/pipeline/FaultCoreUpscaling.jl"))
const U = FaultCoreUpscaling
const F = U.Flow
include(joinpath(cfg["original_code"],"StrictProbe.jl"))
include("AccurateAudit.jl")
const run = TOML.parsefile(joinpath(cfg["science"],"runtime.toml"))
@assert string(VERSION)=="1.12.6" && Threads.nthreads()==2
@assert bytes2hex(open(sha256,run["bridge"]))==run["bridge_sha256"]
BLAS.set_num_threads(2)
F.bridge!(run["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,run["pipeline_hash"])

function write_toml(path, value)
    @assert !ispath(path)
    open(path,"w") do io
        TOML.print(io,value);flush(io)
        @assert ccall(:fsync,Cint,(Cint,),fd(io))==0
    end
end

function big_pressure(x)
    setprecision(BigFloat,384) do
        x isa AbstractMatrix ? [BigFloat(x[k,1])+BigFloat(x[k,2]) for k in axes(x,1)] : BigFloat.(x)
    end
end

function final_checks(p, result, tight)
    x=big_pressure(result.pressure)
    ia=AccurateAudit.independent(p,x)
    original=F.audit(p,AccurateAudit.pair(x))
    oldmass=norm(original[1],1)/max(abs(original[2][3]),floatmin(Float64))
    diff=abs(ia.metrics[3]-result.q)/max(abs(ia.metrics[3]),abs(result.q),floatmin(Float64))
    Dict("independent_passed"=>StrictProbe.reached(ia.metrics,ia.mass,tight) && F.gates(ia.metrics) && diff<=1e-5,
         "independent_metrics"=>ia.metrics,"independent_mass"=>ia.mass,"independent_q_difference"=>diff,
         "original_binary128_passed"=>StrictProbe.reached(original[2],oldmass,tight),
         "original_binary128_metrics"=>original[2],"original_binary128_mass"=>oldmass)
end

@testset "Frozen gates and actual solver integration" begin
    m=[1e-12,1.,1.,1.,1e-12,1e-12,0.,1.]
    @test StrictProbe.reached(m,1e-12,true)
    @test !StrictProbe.reached(m,2e-11,true)
    @test !StrictProbe.reached(m,2e-9,false)
    for contrast in (1.,1e-8)
        net=F.Net(3,[1 2;2 3],[2.,2.],[2.,2.],[1],[3],[2.],[2.])
        mobility=[1.,contrast,1.]
        p,A,_,keep,_=F.prepare(net,mobility)
        sig=F.parts_hash(p,keep)
        result=AccurateAudit.repair(F,StrictProbe,p,A;shift=1e-14,tight=true,deadline=time()+60)
        @test result.passed
        @test abs(result.q-1/sum(1 ./ mobility))/result.q<=1e-9
        @test final_checks(p,result,true)["independent_passed"]
        @test F.parts_hash(p,keep)==sig
    end
end

if cfg["mode"] == "qualify"
    write_toml(joinpath(cfg["out"],"QUALIFIED.toml"),Dict("passed"=>true,"new_production_solves"=>0))
    exit()
end

function run_experiment()
@assert cfg["mode"] in ("precision","iterations")
@assert bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
p,A,rhs,keep=deserialize(cfg["operator"])
@assert F.parts_hash(p,keep)==cfg["parts_hash"]
@assert p.n==cfg["spanning_cells"]
@assert all(isfinite,p.tt) && all(p.tt.>0)
@assert size(A)==(p.n,p.n) && length(rhs)==p.n
@assert p.scale==1.0 ./ sqrt.(diag(A))
start=time();deadline=start+1800;attempts=Any[];solutions=Any[];chosen=Float64[]
passed=true;failed_stage="";independent_results=Any[]
for tight in (false,true)
    success=false
    shifts=tight ? (1e-14,1e-13,1e-12,1e-15) : (1e-14,1e-13,1e-12)
    for shift in shifts
        shift in chosen && continue
        time()>deadline && break
        trial=time()
        try
            result = if cfg["mode"]=="precision"
                AccurateAudit.repair(F,StrictProbe,p,A;shift,tight,deadline=min(deadline,time()+420),
                                    inner_tol=(shift==1e-15 ? 1e-11 : 1e-9),maxouter=80)
            else
                StrictProbe.repair(F,p,A;shift,tight,deadline=min(deadline,time()+420),
                                   inner_tol=(shift==1e-15 ? 1e-11 : 1e-9),maxouter=240)
            end
            push!(attempts,Dict("shift"=>shift,"tight"=>tight,"passed"=>result.passed,
                                "seconds"=>time()-trial,"history"=>result.history))
            name=(tight ? "reference" : "primary")*"_"*string(shift)
            serialize(joinpath(cfg["out"],name*"_pressure.jls"),result.pressure)
            if result.passed
                checks=final_checks(p,result,tight)
                push!(independent_results,merge(Dict("stage"=>name),checks))
                if !checks["independent_passed"]
                    passed=false;failed_stage="independent_audit";break
                end
                push!(chosen,shift);push!(solutions,result);success=true;break
            end
        catch err
            err isa PosDefException || rethrow()
            push!(attempts,Dict("shift"=>shift,"tight"=>tight,"passed"=>false,
                                "seconds"=>time()-trial,"error"=>sprint(showerror,err)))
        end
    end
    if !success
        passed=false
        isempty(failed_stage) && (failed_stage=tight ? "reference" : "primary")
        break
    end
end
d=Dict{String,Any}("passed"=>passed,"mode"=>cfg["mode"],"key"=>cfg["key"],
    "source"=>cfg["source"],"parts_hash"=>cfg["parts_hash"],"operator_sha256"=>cfg["operator_sha256"],
    "seconds"=>time()-start,"attempts"=>attempts,"independent_checks"=>independent_results,
    "failed_stage"=>failed_stage,"scientific_acceptance"=>false,"diagnostic_only"=>true,
    "publication_authorized"=>false,"production_writes"=>0,"completed_phase_calls_repeated"=>0,
    "primary_target"=>1e-9,"reference_target"=>1e-11,"agreement_target"=>1e-5)
if passed
    q1,q2=solutions[1].q,solutions[2].q
    difference=abs(q1-q2)/max(abs(q1),abs(q2),floatmin(Float64))
    d["primary_q"]=q1;d["reference_q"]=q2;d["difference"]=difference;d["shifts"]=chosen
    d["passed"]=difference<=1e-5
    d["failed_stage"]=difference<=1e-5 ? "" : "flow_agreement"
end
@assert F.parts_hash(p,keep)==cfg["parts_hash"]
@assert bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
write_toml(joinpath(cfg["out"],"RESULT.toml"),d)
println("DIAGNOSTIC_RESULT passed=",d["passed"]," mode=",cfg["mode"]," seconds=",d["seconds"])
end
run_experiment()
