using JutulDarcy, Jutul, ForwardDiff, Test, TOML
include("endpoint_check.jl")
@testset "Exact source-derived gas endpoint checks" begin
    t = [0.0 0.0 1.0 1.0; 0.06812057745110492 0.003540380684687808 0.8 2.0; 0.8 0.5 0.0 3.0]
    before = copy(t)
    f = JutulDarcy.PhaseRelativePermeability(t[:,1], t[:,2])
    d = endpoint_evidence(f,t)
    @test d["raw_zero_preserved"] && t == before
    @test d["native_krg_zero"] == d["predicted_native_krg_zero"]
    changed = copy(t); changed[1,2] = 1e-18
    @test_throws AssertionError endpoint_evidence(f,changed)
    changed = copy(t); changed[2,2] += 1e-10
    @test_throws AssertionError endpoint_evidence(f,changed)
    changed = copy(t); changed[2,2] = NaN
    @test_throws AssertionError endpoint_evidence(f,changed)
    wrong = JutulDarcy.PhaseRelativePermeability(t[:,1],t[:,2];epsilon=1e-15)
    @test_throws AssertionError endpoint_evidence(wrong,t)
    wrong = JutulDarcy.PhaseRelativePermeability(t[:,1],t[:,2];epsilon=0.0)
    @test_throws AssertionError endpoint_evidence(wrong,t)
    sharp = [0.0 0.0 1.0 1.0; 1e-12 0.1 0.5 2.0; 0.8 0.5 0.0 3.0]
    wrong = JutulDarcy.PhaseRelativePermeability(sharp[:,1],sharp[:,2])
    @test_throws AssertionError endpoint_evidence(wrong,sharp)
    steep = [0.0 0.0 1.0 1.0; 0.08 0.9 0.5 2.0; 0.8 1.0 0.0 3.0]
    fsteep = JutulDarcy.PhaseRelativePermeability(steep[:,1],steep[:,2])
    ds = endpoint_evidence(fsteep,steep)
    @test eps(Float64) < ds["native_krg_zero"] < 1e-9
    @test ds["native_krg_zero"] == ds["active_slope"]*ds["saturation_shift"]
    changed = copy(steep);changed[1,2]=1e-18
    @test_throws AssertionError endpoint_evidence(fsteep,changed)
    shifted = JutulDarcy.PhaseRelativePermeability(steep[:,1],steep[:,2];epsilon=1e-15)
    @test_throws AssertionError endpoint_evidence(shifted,steep)
    plateau = [0.0 0.0 1.0 1.0; 0.1 0.0 0.9 2.0; 0.5 0.1 0.0 3.0]
    f0 = JutulDarcy.PhaseRelativePermeability(plateau[:,1],plateau[:,2])
    @test f0(0.0) == 0.0
    @test_throws AssertionError endpoint_evidence(f0,plateau)
end
println("ENDPOINT_REGRESSION_PASS")
