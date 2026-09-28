using Test,LinearAlgebra,Random
include("vendor/pipeline/vendor/kr/PhaseFlow.jl")
const F=PhaseFlow
module Previous
include(joinpath(ARGS[2],"vendor/pipeline/vendor/kr/PhaseFlow.jl"))
end
const O=Previous.PhaseFlow
F.bridge!(ARGS[1]);O.bridge!(ARGS[1]);BLAS.set_num_threads(2)
@testset "Unchanged physical operator and strict limits" begin
    for n in (3,8,19),contrast in (1.,1e-6,1e-12,1e-20)
        e=hcat(collect(1:n-1),collect(2:n))
        args=(n,e,fill(2.,n-1),fill(2.,n-1),[1],[n],[2.],[2.])
        net=F.Net(args...);old=O.Net(args...)
        kr=ones(n);kr[cld(n,2)]=contrast
        p,A,b,keep,_=F.prepare(net,kr);po,Ao,bo,ko,_=O.prepare(old,kr)
        @test A==Ao && b==bo && keep==ko
        @test F.parts_hash(p,keep)==O.parts_hash(po,ko)
        q,info=F.conductance(net,kr;force_repair=true)
        exact=1/sum(1 ./kr)
        @test abs(q/exact-1)<=1e-9
        @test info["repair_shifts"][1]!=info["repair_shifts"][2]
        @test info["repair_difference"]<=1e-5
        h=last(info["repair_history"]["reference"])
        @test h[2]<=1e-11 && maximum(h[[6,7,10]])<=1e-11
        @test h[8]>=-1e-9 && h[9]<=1+1e-9
        kr[end]=0
        @test first(F.conductance(net,kr))==0.0
    end
end
@testset "Unchanged successful direct path" begin
    Random.seed!(62049)
    for trial in 1:30
        n=12;e=hcat(collect(1:n-1),collect(2:n))
        args=(n,e,rand(n-1).+0.1,rand(n-1).+0.1,[1],[n],[1.],[1.])
        net=F.Net(args...);old=O.Net(args...);kr=rand(n).+0.01;kr ./=maximum(kr)
        q,a=F.conductance(net,kr);qo,b=O.conductance(old,kr)
        @test !a["fallback_used"] && !b["fallback_used"]
        @test reinterpret(UInt64,q)==reinterpret(UInt64,qo)
        @test a["metrics"]==b["metrics"]
    end
end
