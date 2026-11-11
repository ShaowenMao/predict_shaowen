using TOML,SHA,Serialization,LinearAlgebra
const cfg=TOML.parsefile(ARGS[1])
include(joinpath(cfg["science"],"vendor/pipeline/FaultCoreUpscaling.jl"))
const F=FaultCoreUpscaling.Flow
include("DirectAudit.jl");include("StrictProbe.jl")
@assert string(VERSION)=="1.12.6" && Threads.nthreads()==2
function durable(path,d)
    @assert !ispath(path)
    open(path,"w") do io;TOML.print(io,d);flush(io);@assert ccall(:fsync,Cint,(Cint,),fd(io))==0;end
end
if cfg["mode"]=="qualify"
    durable(joinpath(cfg["out"],"QUALIFIED.toml"),Dict("passed"=>true,"new_production_solves"=>0));exit()
end
@assert cfg["mode"]=="direct"
@assert bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
p,A,rhs,keep=deserialize(cfg["operator"])
@assert F.parts_hash(p,keep)==cfg["parts_hash"] && p.n==cfg["spanning_cells"]
attempts=Any[];qs=Float64[];start=time()
for reference in (false,true)
    d=DirectAudit.solve(p;bits=reference ? 384 : 256,reference)
    name=reference ? "reference" : "primary"
    serialize(joinpath(cfg["out"],name*"_pressure.jls"),d.pressure)
    passed=StrictProbe.reached(d.metrics,d.mass,reference) && F.gates(d.metrics) &&
           StrictProbe.reached(d.independent.metrics,d.independent.mass,reference) && F.gates(d.independent.metrics)
    difference=abs(d.metrics[3]-d.independent.metrics[3])/max(abs(d.metrics[3]),abs(d.independent.metrics[3]))
    passed=passed && difference<=1e-5
    push!(attempts,Dict("passed"=>passed,"reference"=>reference,"bits"=>d.bits,"seconds"=>d.seconds,
        "metrics"=>d.metrics,"mass"=>d.mass,"independent_metrics"=>d.independent.metrics,
        "independent_mass"=>d.independent.mass,"independent_difference"=>difference,
        "order_sha256"=>d.order_sha256,"factor_nonzeros"=>d.factor_nonzeros,"minimum_pivot"=>d.minimum_pivot))
    durable(joinpath(cfg["out"],name*"_AUDIT.toml"),attempts[end])
    push!(qs,d.metrics[3]);passed || break
end
passed=length(attempts)==2 && all(a["passed"] for a in attempts)
difference=length(qs)==2 ? abs(qs[1]-qs[2])/max(abs(qs[1]),abs(qs[2])) : Inf
passed=passed && difference<=1e-5
@assert F.parts_hash(p,keep)==cfg["parts_hash"]
@assert bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
durable(joinpath(cfg["out"],"RESULT.toml"),Dict("passed"=>passed,"mode"=>"direct","key"=>cfg["key"],
    "source"=>cfg["source"],"parts_hash"=>cfg["parts_hash"],"operator_sha256"=>cfg["operator_sha256"],
    "seconds"=>time()-start,"attempts"=>attempts,"difference"=>difference,"flows"=>qs,
    "completed_phase_calls_repeated"=>0,"production_writes"=>0,"scientific_acceptance"=>false,
    "primary_target"=>1e-9,"reference_target"=>1e-11,"agreement_target"=>1e-5,
    "regularization"=>false,"changed_conductances"=>false,"diagnostic_only"=>true))
println("DIRECT_DIAGNOSTIC passed=",passed," seconds=",time()-start)
