using Test,SparseArrays
include("PrecisionRecovery.jl")
@testset "Dense and sparse pressure identity" begin
    for bits in (256,384,512)
        x=setprecision(BigFloat,bits) do;[BigFloat(0.125),BigFloat(0.75)];end
        for input in (x,sparse(x))
            y=PrecisionRecovery.pressure_vector(input,2,bits>=384)
            @test y isa Vector{BigFloat}
            @test isequal(y,x)
            @test precision.(y)==precision.(x)
        end
    end
    @test_throws AssertionError PrecisionRecovery.pressure_vector([.1,.2],2,false)
    @test_throws AssertionError PrecisionRecovery.pressure_vector(BigFloat[.1],2,false)
    @test_throws AssertionError PrecisionRecovery.pressure_vector(BigFloat[.1],1,true)
end
