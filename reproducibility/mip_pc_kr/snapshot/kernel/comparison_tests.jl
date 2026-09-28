using Test
@testset "Exact microstate and bounded scalar reduction review" begin
    @test same_average(0.0,0.0)
    @test !same_average(0.0,nextfloat(0.0))
    @test same_average(nextfloat(0.5,8),0.5)
    @test !same_average(0.5+1e-12,0.5)
    @test !same_average(NaN,0.5)
    header="sample,pressure_pa,sg_physical,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction,events,state_hash"
    a=vcat([header],["$i,100.0,0.5,0.0,0.0,1,0.5,3,abc" for i in 1:100])
    @test compare_pc_lines(a,a)==0
    b=copy(a);b[2]=replace(b[2],",abc"=>",changed")
    @test_throws AssertionError compare_pc_lines(a,b)
    b=copy(a);b[2]=replace(b[2],",3,abc"=>",4,abc")
    @test_throws AssertionError compare_pc_lines(a,b)
    b=copy(a);b[2]=replace(b[2],"100.0"=>"100.1")
    @test_throws AssertionError compare_pc_lines(a,b)
end
