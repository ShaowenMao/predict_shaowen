using Test,SHA,TOML
include("FaultCoreUpscaling.jl")
using .FaultCoreUpscaling
const U=FaultCoreUpscaling
raw,library,out=ARGS
input=U.Pc.read_input(raw);n=length(input.phi)
@assert input.edges==hcat(collect(1:n-1),collect(2:n))
@assert !ispath(out);mkpath(out)
k=input.perm[:,end]
net=U.Flow.Net(n,input.edges,2k[input.edges[:,1]],2k[input.edges[:,2]],input.inlet,input.outlet,
              2k[input.inlet],2k[input.outlet])
hash=open(sha256,raw);geo=Geometry(input.core,3,Float64(n),1.,hash,net)
write_geometry(joinpath(out,"geometry.bin"),geo)
@testset "Binding, direction and overwrite guards" begin
    g=read_geometry(joinpath(out,"geometry.bin"))
    @test validate_geometry(input,g,hash)
    @test_throws AssertionError validate_geometry(input,g,zeros(UInt8,32))
    @test_throws AssertionError validate_geometry(input,Geometry(g.core,1,g.length,g.area,hash,net),hash)
    @test_throws AssertionError write_geometry(joinpath(out,"geometry.bin"),geo)
    @test_throws AssertionError run_files(raw,joinpath(out,"geometry.bin"),library,out)
end
rows=run_files(raw,joinpath(out,"geometry.bin"),library,joinpath(out,"cached");save_states=true)
other=run_files(raw,joinpath(out,"geometry.bin"),library,joinpath(out,"uncached");cache_states=false)
@testset "Exact cache and analytic series flow" begin
    @test rows==other
    summary=TOML.parsefile(joinpath(out,"cached","summary.toml"))
    @test summary["normalization"]=="fresh Julia intrinsic conductance, same TPFA network and corrected direct solver"
    @test summary["unique_mobility_states"]<=100
    expected_qa=1/sum(1 ./ k)
    @test isapprox(summary["intrinsic_conductance"],expected_qa;rtol=1e-9,atol=0.)
    U.Pc.read_archive(joinpath(out,"cached","states.bin");expected_model_hash=hash) do j,state,kw,kg
        for (kr,value) in ((kw,rows[j].krw),(kg,rows[j].krg))
            q=any(==(0),kr) ? 0. : 1/sum(1 ./ (k.*kr))
            @test (value==0)==(q==0)
            @test isapprox(value,q/expected_qa;rtol=1e-5,atol=0.)
        end
    end
    @test isfile(joinpath(out,"cached","completion.toml"))
    @test !isfile(joinpath(out,"uncached","states.bin"))
end
