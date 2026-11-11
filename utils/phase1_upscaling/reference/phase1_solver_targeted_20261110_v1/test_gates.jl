using Test
include("StrictProbe.jl")
@testset "Strict targets unchanged" begin
    m=[1e-12,1.,1.,1.,1e-12,1e-12,0.,1.]
    @test StrictProbe.reached(m,1e-12,true)
    @test StrictProbe.reached(m,1e-9,false)
    @test !StrictProbe.reached(m,1.01e-9,false)
    @test !StrictProbe.reached(m,1.01e-11,true)
    @test !StrictProbe.reached(m,NaN,false)
    for i in (1,5,6)
        b=copy(m);b[i]=1e-7
        @test !StrictProbe.reached(b,0.,false)
    end
    for (i,v) in ((3,0.),(7,-2e-9),(8,1+2e-9),(2,Inf))
        b=copy(m);b[i]=v
        @test !StrictProbe.reached(b,0.,false)
    end
end
