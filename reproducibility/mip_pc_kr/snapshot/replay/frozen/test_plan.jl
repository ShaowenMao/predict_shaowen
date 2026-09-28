using Test,SHA
include("plan_and_run.jl")
@assert length(ARGS)==3
raw,geom,out=ARGS
x=P.read_input(raw);pp,pool=dictionary(x)
@testset "Fixed100 pressure-only plan and binary roundtrip" begin
    @test length(pp)==100
    @test all(diff(pp).>0)
    @test pp[1]==0
    @test pp==P.fixed_from_dictionary(pool,100)
    @test pp[end]>=maximum(first(P.prepare_model(x)).endpoint)
    plan(raw,geom,out);y=P.read_input(joinpath(out,"raw.bin"))
    for f in (:core,:perm,:phi,:volume,:edges,:inlet,:outlet,:material,:tables)
        @test getfield(x,f)==getfield(y,f)
    end
    @test y.pressures==pp && y.dictionary==pool
    @test validate_geometry(y,read_geometry(joinpath(out,"geometry.bin")),open(sha256,joinpath(out,"raw.bin")))
    @test_throws AssertionError plan(raw,geom,out)
end
