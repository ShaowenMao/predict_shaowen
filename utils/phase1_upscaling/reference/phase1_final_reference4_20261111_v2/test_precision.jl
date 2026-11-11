using Test,LinearAlgebra,SparseArrays,Random
include("DirectAudit.jl");include("StrictProbe.jl")

function chain(n,weak)
    t=fill(2.,n-1);t[div(n,2)]=weak
    (;n,i=collect(1:n-1),j=collect(2:n),tt=t,ii=[1],oo=[n],tin=[2.],tout=[2.])
end

@testset "Exact direct arithmetic and unchanged gates" begin
    for n in (3,9,87),weak in (1.,1e-8,1e-24,1e-40)
        p=chain(n,weak)
        expected=setprecision(BigFloat,512) do;inv(1+sum(inv,BigFloat.(p.tt)));end
        for reference in (false,true)
            d=DirectAudit.solve(p;bits=reference ? 384 : 256,reference)
            @test StrictProbe.reached(d.metrics,d.mass,reference)
            @test StrictProbe.reached(d.independent.metrics,d.independent.mass,reference)
            @test abs(d.metrics[3]-expected)/expected<1e-12
            @test d.minimum_pivot>0 && d.new_solves==1
        end
    end
    Random.seed!(715)
    for n in (8,20)
        i=collect(1:n-1);j=collect(2:n);append!(i,ones(Int,n-2));append!(j,collect(3:n))
        p=(;n,i,j,tt=10. .^ (-24rand(length(i))),ii=[1,2],oo=[n],tin=[1.,1e-20],tout=[1e-10])
        d=DirectAudit.solve(p)
        expected=setprecision(BigFloat,512) do;A,b=DirectAudit.assemble(p);Matrix(A)\b;end
        @test maximum(abs.(d.pressure.-expected))<big"1e-40"
        @test StrictProbe.reached(d.metrics,d.mass,true)
    end
    @test_throws AssertionError DirectAudit.solve(chain(9,1e-24);maxfill=1)
    @test_throws AssertionError DirectAudit.solve(chain(3,1.);bits=64)
    m=[1e-12,1.,1.,1.,1e-12,1e-12,0.,1.]
    @test !StrictProbe.reached(m,2e-11,true)
    @test !StrictProbe.reached(m,2e-9,false)
end
