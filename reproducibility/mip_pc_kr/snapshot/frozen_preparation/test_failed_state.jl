using Test,LinearAlgebra
include("vendor/pipeline/FaultCoreUpscaling.jl")
const U=FaultCoreUpscaling
raw,geometry,bridge,sample_arg,phase=ARGS
input=U.Pc.read_input(raw);g=U.read_geometry(geometry)
U.validate_geometry(input,g,U.filehash(raw));U.Flow.bridge!(bridge);BLAS.set_num_threads(2)
m,_=U.Pc.prepare_model(input);paths=U.Pc.prepare_paths(m)
s=U.Pc.initial(m.n);w=U.Pc.Workspace(m.n);sample=parse(Int,sample_arg)
for j in 1:sample
    U.Pc.advance_pc!(m,s,m.pressures[j],w,paths);U.Pc.observe(m,s,w)
end
q,info=U.Flow.conductance(g.net,phase=="brine" ? w.kw : w.kg)
@testset "Formerly failing frozen state" begin
    @test q>0 && isfinite(q)
    @test info["fallback_used"]
    @test U.Flow.gates(info["metrics"])
    @test info["repair_difference"]<=1e-5
    h=last(info["repair_history"]["reference"])
    @test h[2]<=1e-11 && maximum(h[[6,7,10]])<=1e-11
    @test h[8]>=-1e-9 && h[9]<=1+1e-9
end
println("RECOVERED_STATE core=",input.core," sample=",sample," q=",q," shifts=",info["repair_shifts"],
    " difference=",info["repair_difference"]," metrics=",info["metrics"])
