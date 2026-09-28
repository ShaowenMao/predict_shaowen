using Test,SHA
include("plan_and_run.jl")
@assert length(ARGS)==3
raw,geom,out=ARGS
x=P.read_input(raw);pp,pool=dictionary(x)
@testset "Fixed100 pressure-only plan and binary roundtrip" begin
    @test length(pp)==100
    @test all(diff(pp).>0)
    @test pp[1]==0
    model,_=P.prepare_model(x)
    @test pp==cover_endpoint(P.fixed_from_dictionary(pool,100),pool,maximum(model.endpoint),max(maximum(model.endpoint),maximum(x.dictionary;init=0.)))
    @test pp[end]>=maximum(first(P.prepare_model(x)).endpoint)
    plan(raw,geom,out);y=P.read_input(joinpath(out,"raw.bin"))
    for f in (:core,:perm,:phi,:volume,:edges,:inlet,:outlet,:material,:tables)
        @test getfield(x,f)==getfield(y,f)
    end
    @test y.pressures==pp && y.dictionary==pool
    @test validate_geometry(y,read_geometry(joinpath(out,"geometry.bin")),open(sha256,joinpath(out,"raw.bin")))
    @test_throws AssertionError plan(raw,geom,out)
end

@testset "Exact endpoint, without loosening the coverage guard" begin
    good=collect(range(0.,100.;length=100))
    @test cover_endpoint(good,good,100.,100.)==good
    @test good[end]==100.
    for n in 1:16
        old=copy(good);old[end]=prevfloat(100.,n)
        pool=sort(unique(vcat(old,100.)))
        fixed=cover_endpoint(old,pool,100.,100.)
        @test fixed[end]===100.
        @test fixed[1:end-1]==old[1:end-1]
        @test old[end]===prevfloat(100.,n)
    end
    bad=copy(good);bad[end]=prevfloat(100.,17)
    @test_throws AssertionError cover_endpoint(bad,sort(vcat(bad,100.)),100.,100.)
    @test_throws AssertionError cover_endpoint(good[1:99],good,100.,100.)
    @test_throws AssertionError cover_endpoint(good,good,101.,101.)
    @test_throws AssertionError cover_endpoint(good,good,NaN,100.)
    over=copy(good);over[end]=nextfloat(100.)
    @test cover_endpoint(over,sort(vcat(over,100.)),100.,100.)==over
end
