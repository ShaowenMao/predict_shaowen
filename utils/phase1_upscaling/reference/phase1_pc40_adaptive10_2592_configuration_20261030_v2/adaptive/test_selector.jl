using Test, TOML
include("AdaptiveSelector.jl")
using .AdaptiveSelector: select_adaptive
@testset "adaptive ten-query contract" begin
    s=collect(range(0.0,1.0,length=100)); calls=Int[]
    kr=hcat(1 .- s,s)
    query(i)=(push!(calls,i); kr[i,:])
    ids,order=select_adaptive(s,query)
    @test length(ids)==length(unique(calls))==10
    @test ids==sort(calls) && order==calls
    @test first(ids)==1 && last(ids)==100
    kr[setdiff(1:100,ids),:] .= .9
    empty!(calls)
    again,_=select_adaptive(s,query)
    @test again==ids
    k=Float64.(collect(1:100) .>= 3)
    steps,_=select_adaptive(s,i->[1-k[i],k[i]])
    @test 2 in steps && 3 in steps
    @test first(select_adaptive([0.,0.,.1,.1,.9,.9],i->[0.,1.]))==[1,3,6]
    @test first(select_adaptive(zeros(100),i->[0.,0.]))==[1,100]
    @test_throws AssertionError select_adaptive([1.,0.],i->[0.,0.])
    @test_throws AssertionError select_adaptive([0.,1.],i->[NaN,0.])
    @test_throws AssertionError select_adaptive([0.,1.],i->[0.,0.],budget=1)
    fixture=TOML.parsefile(joinpath(@__DIR__,"fixtures.toml"))
    for c in fixture["cases"]
        selected,ordered=select_adaptive(c["s"],i->[c["krw"][i],c["krg"][i]])
        @test selected==c["expected"]
        @test ordered==c["query_order"]
    end
end
