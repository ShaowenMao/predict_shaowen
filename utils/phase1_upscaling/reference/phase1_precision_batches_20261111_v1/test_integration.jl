using TOML,SHA,Serialization,LinearAlgebra,SparseArrays,Test
cfg=TOML.parsefile(ARGS[1])
include(joinpath(cfg["science"],"vendor/pipeline/FaultCoreUpscaling.jl"))
const F=FaultCoreUpscaling.Flow
include("PrecisionRecovery.jl")
R=PrecisionRecovery
run=TOML.parsefile(joinpath(cfg["science"],"runtime.toml"))
@assert VERSION==v"1.12.6" && Threads.nthreads()==2
@assert bytes2hex(open(sha256,run["bridge"]))==run["bridge_sha256"]
F.bridge!(run["bridge"]);BLAS.set_num_threads(2)
FaultCoreUpscaling.Diagnostics.verify_receipt!(FaultCoreUpscaling._source_hash_impl,run["pipeline_hash"])
@testset "Accurate full-recovery path and saved-pressure reuse" begin
    for (index,contrast) in enumerate((1.,1e-8))
        net=F.Net(3,[1 2;2 3],[2.,2.],[2.,2.],[1],[3],[2.],[2.])
        k=[1.,contrast,1.];p,A,_,keep,_=F.prepare(net,k)
        signature=F.parts_hash(p,keep)
        result=R.two_checks(F,p,A;deadline=time()+60,out=cfg["out"],prefix="synthetic$(index)")
        @test result["passed"]
        @test abs(result["reference_q"]-1/sum(1 ./ k))/result["reference_q"]<=1e-9
        @test result["shifts"][1]!=result["shifts"][2]
        @test all(a["passed"] for a in result["audits"])
        @test F.parts_hash(p,keep)==signature
        @test !R.checks(F,p,fill(BigFloat(.5),p.n),1.,true)["passed"]
        root=joinpath(cfg["out"],"reuse$(index)");mkpath(root)
        for (j,tight) in enumerate((false,true))
            attempt=filter(a->a["passed"] && a["tight"]==tight,result["attempts"])[1]
            cp(joinpath(cfg["out"],attempt["pressure_file"]),joinpath(root,(tight ? "reference_" : "primary_")*string(attempt["shift"])*"_pressure.jls"))
        end
        result["mode"]="precision";result["parts_hash"]=signature
        open(joinpath(root,"RESULT.toml"),"w") do io;TOML.print(io,result);end
        saved=Dict("precision"=>root,"probe_sha256"=>bytes2hex(open(sha256,joinpath(root,"RESULT.toml"))),
                   "parts_hash"=>signature,"probe_q"=>result["reference_q"])
        proof=R.reaudit(F,p,saved)
        @test proof["passed"] && proof["new_solves"]==0
        wrong=copy(saved);wrong["parts_hash"]="bad"
        @test_throws AssertionError R.reaudit(F,p,wrong)
        wrong=copy(saved);wrong["probe_q"]*=2
        @test_throws AssertionError R.reaudit(F,p,wrong)
    end
end
open(joinpath(cfg["out"],"QUALIFIED.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"production_solves"=>0,"saved_result_reuse_tested"=>true))
end
