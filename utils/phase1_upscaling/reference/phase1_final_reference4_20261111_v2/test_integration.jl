using TOML,SHA,Serialization,LinearAlgebra,SparseArrays,Test
cfg=TOML.parsefile(ARGS[1])
include(joinpath(cfg["science"],"vendor/pipeline/FaultCoreUpscaling.jl"))
const F=FaultCoreUpscaling.Flow
include("PrecisionRecovery.jl")
include("SavedPrimary.jl")
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
        @test result["attempts"][1]["order_sha256"]!=result["attempts"][2]["order_sha256"]
        @test all(a["passed"] for a in result["audits"])
        @test F.parts_hash(p,keep)==signature
        @test !R.checks(F,p,fill(BigFloat(.5;precision=384),p.n),1.,true)["passed"]
        root=joinpath(cfg["out"],"reuse$(index)");mkpath(root)
        for (j,tight) in enumerate((false,true))
            attempt=filter(a->a["passed"] && a["reference"]==tight,result["attempts"])[1]
            cp(joinpath(cfg["out"],attempt["pressure_file"]),joinpath(root,(tight ? "reference" : "primary")*"_pressure.jls"))
        end
        result["mode"]="direct";result["parts_hash"]=signature
        open(joinpath(root,"RESULT.toml"),"w") do io;TOML.print(io,result);end
        saved=Dict("precision"=>root,"probe_sha256"=>bytes2hex(open(sha256,joinpath(root,"RESULT.toml"))),
                   "parts_hash"=>signature,"probe_q"=>result["reference_q"])
        proof=R.reaudit(F,p,saved)
        @test proof["passed"] && proof["new_solves"]==0
        x=deserialize(joinpath(root,"reference_pressure.jls"))
        @test R.checks(F,p,sparse(x),result["reference_q"],true)["passed"]
        @test R.checks(F,p,collect(x),result["reference_q"],true)["passed"]
        wrong=copy(saved);wrong["parts_hash"]="bad"
        @test_throws AssertionError R.reaudit(F,p,wrong)
        wrong=copy(saved);wrong["probe_q"]*=2
        @test_throws AssertionError R.reaudit(F,p,wrong)
    end
end
@testset "Reference-only continuation of saved primary" begin
    for (j,contrast) in enumerate((1.,1e-8))
        net=F.Net(3,[1 2;2 3],[2.,2.],[2.,2.],[1],[3],[2.],[2.])
        k=[1.,contrast,1.];p,A,_,keep,_=F.prepare(net,k);signature=F.parts_hash(p,keep)
        primary=R.DirectAudit.solve(p;bits=256,reference=false)
        for paired in (false,true)
            x=paired ? R.AccurateAudit.pair(primary.pressure) : primary.pressure
            root=joinpath(cfg["out"],"single$(j)_$(paired)");mkpath(root)
            file=joinpath(root,"saved.jls");serialize(file,x);hash=bytes2hex(open(sha256,file));count=Ref(0)
            function reference(p;bits,reference)
                @test bits==384 && reference
                count[]+=1;R.DirectAudit.solve(p;bits,reference)
            end
            d=SavedPrimary.reference_only(F,p;primary=file,expected_hash=hash,original_q=primary.metrics[3],out=root,solve_reference=reference)
            @test d["passed"] && count[]==1 && d["new_primary_solves"]==0 && d["new_reference_solves"]==1
            @test F.parts_hash(p,keep)==signature && bytes2hex(open(sha256,file))==hash
            d["mode"]="saved_primary_direct_reference";d["parts_hash"]=signature
            path=joinpath(root,"RESULT.toml");SavedPrimary.durable(path,d)
            saved=Dict("precision"=>root,"parts_hash"=>signature,"probe_sha256"=>bytes2hex(open(sha256,path)),"probe_q"=>d["flows"][2])
            audit=SavedPrimary.reaudit(F,p,saved)
            @test audit["passed"] && audit["new_solves"]==0 && count[]==1
            @test_throws AssertionError SavedPrimary.reference_only(F,p;primary=file,expected_hash="bad",original_q=primary.metrics[3],out=root,solve_reference=reference)
            @test count[]==1
            bad=joinpath(root,"bad.jls");serialize(bad,fill(BigFloat(.5;precision=256),p.n))
            @test_throws AssertionError SavedPrimary.reference_only(F,p;primary=bad,expected_hash=bytes2hex(open(sha256,bad)),original_q=primary.metrics[3],out=root,solve_reference=reference)
            @test count[]==1
        end
    end
end
open(joinpath(cfg["out"],"QUALIFIED.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"production_solves"=>0,"saved_result_reuse_tested"=>true))
end
