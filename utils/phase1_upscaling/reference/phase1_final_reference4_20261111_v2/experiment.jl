using TOML,SHA,Serialization,LinearAlgebra
const cfg=TOML.parsefile(ARGS[1])
include(joinpath(cfg["science"],"vendor/pipeline/FaultCoreUpscaling.jl"))
const F=FaultCoreUpscaling.Flow
include("PrecisionRecovery.jl");include("SavedPrimary.jl")
@assert VERSION==v"1.12.6" && Threads.nthreads()==2 && cfg["mode"]=="saved_primary_direct_reference"
run=TOML.parsefile(joinpath(cfg["science"],"runtime.toml"))
@assert bytes2hex(open(sha256,run["bridge"]))==run["bridge_sha256"]
F.bridge!(run["bridge"]);BLAS.set_num_threads(2)
FaultCoreUpscaling.Diagnostics.verify_receipt!(FaultCoreUpscaling._source_hash_impl,run["pipeline_hash"])
@assert bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
p,A,rhs,keep=deserialize(cfg["operator"])
@assert F.parts_hash(p,keep)==cfg["parts_hash"] && p.n==cfg["spanning_cells"]
d=SavedPrimary.reference_only(F,p;primary=cfg["primary"],expected_hash=cfg["primary_sha256"],original_q=cfg["primary_q"],out=cfg["out"])
@assert F.parts_hash(p,keep)==cfg["parts_hash"] && bytes2hex(open(sha256,cfg["operator"]))==cfg["operator_sha256"]
merge!(d,Dict("mode"=>cfg["mode"],"key"=>cfg["key"],"source"=>cfg["source"],"parts_hash"=>cfg["parts_hash"],
    "operator_sha256"=>cfg["operator_sha256"],"primary_source"=>cfg["primary_source"],"primary_result_sha256"=>cfg["primary_result_sha256"],
    "completed_phase_calls_repeated"=>0,"production_writes"=>0,"scientific_acceptance"=>false,"diagnostic_only"=>true,
    "regularization"=>false,"changed_conductances"=>false))
SavedPrimary.durable(joinpath(cfg["out"],"RESULT.toml"),d)
println("REFERENCE_ONLY_DIAGNOSTIC passed=",d["passed"])
