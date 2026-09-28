using Test, JutulDarcy, Jutul, ForwardDiff
root = ENV["CORRECTED_STUDY_ROOT"]
runtime = joinpath(root, "runtime")
@testset "Existing relative-permeability regression tests" begin
    include(joinpath(runtime,"test","relperm.jl"))
end
@testset "Existing inventory and QoI regression tests" begin
    include(joinpath(runtime,"test","production_qoi.jl"))
end
@testset "Explicit per-region drainage rule" begin
    sg = [0., .01, .1, .3, .6, 1.]
    table = hcat(sg, sg.^2, (1 .- sg).^3, 1e5 .* sg)
    props = Dict{String,Any}("SGOF"=>[table,table,table,table],
        "JUTUL_DRAINAGE_ONLY_REGIONS"=>[2])
    kr = JutulDarcy.deck_relperm(Dict(),props;oil=true,water=false,gas=true,satnum=[1,2])
    @test !JutulDarcy.region_is_drainage_only(kr,1)
    @test JutulDarcy.region_is_drainage_only(kr,2)
    @test !JutulDarcy.region_is_drainage_only(kr,3)
    @test !JutulDarcy.region_is_drainage_only(kr,0)
    @test_throws ErrorException JutulDarcy.ReservoirRelativePermeabilities(
        g=kr.krg,og=kr.krog,regions=[1,2],drainage_only_regions=[0])
    ac = JutulDarcy.ProductionQoIGasMobilityAccounting(:killough,[.01,.01],
        [.2,.2],[.9,.9],[.9,.9],0.1,.05,0.,1e-10,BitVector([false,true]))
    host = JutulDarcy.production_qoi_active_gas_mobility(ac,1,.1,.9)
    fault = JutulDarcy.production_qoi_active_gas_mobility(ac,2,.1,.9)
    @test host.history_dependent && host.branch == :imbibition
    @test fault == (critical=.01,branch=:drainage,history_dependent=false)
end
println("CORRECTED_HYSTERESIS_UNIT_TESTS_PASS")
