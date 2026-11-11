using Test,Serialization,SparseArrays
include("PrecisionRecovery.jl");include("SavedPrimary.jl")
@testset "Exact saved pressure representation, no solves" begin
    x=[1.0 2.0^-80;0.5 -2.0^-80;0.0 0.0]
    y=SavedPrimary.exact_vector(x,3)
    @test all(precision.(y).==384)
    @test y[1]>1 && y[2]<.5 && y[3]==0
    @test SavedPrimary.exact_vector(sparse(y),3)==y
    @test_throws AssertionError SavedPrimary.exact_vector(x,2)
    bad=copy(x);bad[1,1]=NaN
    @test_throws AssertionError SavedPrimary.exact_vector(bad,3)
    @test_throws AssertionError SavedPrimary.exact_vector(fill(BigFloat(1;precision=128),3),3)
    slots=VERSION==v"1.12.6" ? ("096","192","221","229") : ()
    isempty(slots) && @info "Actual Julia1.12.6 serialized fixtures are mandatory in compute qualification"
    for slot in slots
        p=joinpath(@__DIR__,"fixtures",slot*"_pressure.jls")
        raw=deserialize(p);v=SavedPrimary.exact_vector(raw,size(raw,1))
        @test all(isfinite,v) && all(precision.(v).>=256)
        if raw isa AbstractVector
            @test v==raw
        else
            exact=setprecision(BigFloat,2048) do
                [BigFloat(raw[k,1])+BigFloat(raw[k,2]) for k in axes(raw,1)]
            end
            @test v==exact
        end
    end
end
