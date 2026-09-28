using Test, SHA, LinearAlgebra
const snapshot = joinpath(@__DIR__, "..", "snapshot")
include(joinpath(snapshot, "pipeline", "FaultCoreUpscaling.jl"))
include(joinpath(snapshot, "backend", "Backend.jl"))
include(joinpath(snapshot, "kernel", "SelectStates.jl"))
const U = FaultCoreUpscaling
BLAS.set_num_threads(2)
raw, geometry, bridge = ARGS
input = U.Pc.read_input(raw)
g = U.read_geometry(geometry)
@assert U.validate_geometry(input, g, open(sha256, raw))
U.Flow.bridge!(bridge)
Backend.reset!()
solver = Backend.Solver(U.Flow)
@testset "Managed factors retain corrected-flow results" begin
    try
        for fraction in (1., .8, .5, .1, .01), phase in ("brine", "gas")
            kr = fill(fraction, g.net.n)
            expected, _ = U.Flow.conductance(g.net, kr)
            observed, _ = solver(g.net, kr, phase)
            @test isapprox(observed, expected; rtol=1e-10, atol=0.)
        end
    finally
        Backend.clear!(solver)
    end
    receipt = Backend.report()
    @test receipt["numeric_created"] == receipt["numeric_released"]
    @test receipt["symbolic_created"] == receipt["symbolic_released"]
end
@testset "Sparse sampling retains endpoints and unique states" begin
    indices, targets = SelectStates.select_states(collect(range(0., .8; length=100)), 10)
    @test length(indices) == 10
    @test first(indices) == 1 && last(indices) == 100
    @test issorted(indices) && length(unique(indices)) == 10
end
