using Test, LinearAlgebra, SparseArrays
include("AccurateAudit.jl")

@testset "High-precision conservative audit" begin
    p = (;n=2, i=[1.], j=[2.], tt=[1.], ii=[1.], oo=[2.], tin=[1.], tout=[1.])
    x = setprecision(BigFloat,256) do; [BigFloat(2)/3, BigFloat(1)/3]; end
    a=AccurateAudit.audit(p,x); b=AccurateAudit.independent(p,x)
    @test a.mass < 1e-60 && b.mass < 1e-60
    @test a.metrics[3] == b.metrics[3] == 1/3
    @test maximum(a.metrics[5:6]) < 1e-60
    @test maximum(b.metrics[5:6]) < 1e-60
    bad=AccurateAudit.audit(p,BigFloat[.5,.5])
    @test bad.mass > .1
    @test bad.metrics[6] == 0 # Global energy agreement does not prove cellwise balance.
    # A small pressure correction near one must not disappear in reconstruction.
    y=setprecision(BigFloat,256) do; [BigFloat(1)-BigFloat(2)^(-70)]; end
    delta=2.0^(-121); z=AccurateAudit.add(y,[delta])
    @test z[1]-y[1] == BigFloat(delta)
    h=AccurateAudit.pair(z)
    @test BigFloat(h[1,1])+BigFloat(h[1,2]) == z[1]
    # Audit two parallel outlet connections and a near-zero flow without clipping.
    small=(;n=1,i=Float64[],j=Float64[],tt=Float64[],ii=[1.],oo=[1.,1.],
             tin=[1e-30],tout=[.5,.5])
    sx=setprecision(BigFloat,256) do; [BigFloat(small.tin[1])/(1+BigFloat(small.tin[1]))]; end
    aa=AccurateAudit.audit(small,sx); bb=AccurateAudit.independent(small,sx)
    @test aa.metrics[3] > 0 && bb.metrics[3] > 0
    @test aa.mass < 1e-60 && bb.mass < 1e-60
    @test maximum(bb.metrics[5:6]) < 1e-60
end
