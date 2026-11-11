using Test
include("StrictProbe.jl")
struct ExtraFixture end
const ATTEMPTS=Tuple{Bool,Float64,Float64}[]
function StrictProbe.repair(flow,p::ExtraFixture,A;shift,tight,deadline,inner_tol=1e-9,maxouter=80)
    push!(ATTEMPTS,(tight,shift,inner_tol))
    good=(!tight && shift==1e-14)||(tight && shift==1e-15)
    h=[1.,1e-13,1.,1.,1.,1e-13,1e-13,0.,1.,1e-13]
    (;passed=good,q=1.,metrics=h[2:9],history=[h])
end
@testset "Additional reference effort, unchanged primary and gates" begin
    empty!(ATTEMPTS)
    d=StrictProbe.two_checks(nothing,ExtraFixture(),nothing;deadline=time()+60)
    @test !d["passed"] && d["failed_stage"]=="reference"
    @test all(a[2]!=1e-15 for a in ATTEMPTS)
    empty!(ATTEMPTS)
    d=StrictProbe.two_checks(nothing,ExtraFixture(),nothing;deadline=time()+60,reference_extra=true)
    @test d["passed"] && d["difference"]==0
    @test d["shifts"]==[1e-14,1e-15]
    @test ATTEMPTS[end]==(true,1e-15,1e-11)
    @test count(!a[1] for a in ATTEMPTS)==1
end
