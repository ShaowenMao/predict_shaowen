using Test
include("ResumeContract.jl")
@testset "Exact saved prefix, no blind recalculation" begin
    rows=[Dict("sample"=>0,"phase"=>"intrinsic","q"=>1.),Dict("sample"=>57,"phase"=>"brine","q"=>.5)]
    l=ResumeContract.Ledger(rows)
    @test_throws AssertionError ResumeContract.take_saved!(l,57,"brine")
    @test l.next==1
    @test ResumeContract.take_saved!(l,0,"intrinsic")==rows[1]
    @test_throws AssertionError ResumeContract.take_saved!(l,57,"gas")
    @test !ResumeContract.finished(l)
    @test ResumeContract.take_saved!(l,57,"brine")==rows[2]
    @test ResumeContract.finished(l)
    @test ResumeContract.take_saved!(l,57,"gas")===nothing
end
for name in ("recover_flow.jl","StrictProbe.jl")
    ex=Meta.parseall(read(joinpath(@__DIR__,name),String))
    @assert !any(x->x isa Expr && x.head in (:error,:incomplete),ex.args)
end
