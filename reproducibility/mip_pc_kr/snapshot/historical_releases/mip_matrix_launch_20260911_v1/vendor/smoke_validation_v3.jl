using JutulDarcy,Jutul,MAT,SHA,TOML,LinearAlgebra
BLAS.set_num_threads(1)
input,qualified,out,slots_arg=ARGS
@assert !ispath(out)
q=TOML.parsefile(joinpath(qualified,"constitutive_pass.toml"))
@assert q["passed"]&&q["tables"]==522&&q["table_sha256"]==bytes2hex(open(sha256,input))
@assert TOML.parsefile(joinpath(qualified,"full_fault_contract_pass.toml"))["passed"]
mkpath(out)
D=matread(input);T=Matrix{Float64}.(vec(D["exportTables"]));slotmap=Int.(vec(D["tableSlotMap"]))

function smoke(t,sg0)
    n=16;domain=get_1d_reservoir(n)
    sys=ImmiscibleSystem((LiquidPhase(),VaporPhase()))
    model=SimulationModel(domain,sys)
    props=Dict("SGOF"=>[t])
    kr=JutulDarcy.deck_relperm(Dict(),props;oil=true,water=false,gas=true)
    pc=JutulDarcy.deck_pc(props;oil=true,water=false,gas=true)
    replace_variables!(model,RelativePermeabilities=kr)
    set_secondary_variables!(model;CapillaryPressure=pc)
    @assert haskey(model.secondary_variables,:CapillaryPressure)
    dt=fill(1000.,10);pv=sum(pore_volume(domain));rate=.02pv/sum(dt)
    src=SourceTerm(1,rate,fractional_flow=[0.,1.],type=JutulDarcy.VolumeSource)
    @assert src.type==JutulDarcy.VolumeSource
    bc=FlowBoundaryCondition(n,3e7,fractional_flow=[1.,0.])
    forces=setup_forces(model,sources=src,bc=bc)
    parameters=setup_parameters(model,PhaseViscosities=[1e-3,5e-5])
    state0=setup_state(model,Pressure=3e7,Saturations=[1-sg0,sg0])
    start=time()
    states,reports=simulate(state0,model,dt;forces=forces,parameters=parameters,
        info_level=-1,max_timestep_cuts=20)
    @assert length(states)==length(dt)
    for s in states
        @assert all(isfinite,s[:Pressure])&&all(s[:Pressure].>0)
        @assert all(isfinite,s[:Saturations])&&minimum(s[:Saturations])>=-1e-9&&maximum(s[:Saturations])<=1+1e-9
        @assert maximum(abs.(sum(s[:Saturations];dims=1).-1))<1e-10
        @assert all(isfinite,pc.pc[1][1].(s[:Saturations][2,:]))
    end
    final=states[end][:Saturations][2,:]
    @assert maximum(abs.(final.-sg0))>1e-8
    Dict("initial_sg"=>sg0,"reported_steps"=>length(states),"completed_time_seconds"=>sum(dt),
        "wall_seconds"=>time()-start,"source_type"=>"VolumeSource (in-situ)",
        "source_rate_m3_s"=>rate,"domain_pore_volume_m3"=>pv,
        "injected_pore_volumes"=>rate*sum(dt)/pv,
        "final_sg_min"=>minimum(final),"final_sg_max"=>maximum(final),
        "final_sg_pv_mean"=>sum(final.*pore_volume(domain))/pv,
        "capillary_pressure_active"=>true,"passed"=>true)
end

results=[]
for slot in parse.(Int,split(slots_arg,','))
    t=T[findfirst(==(slot),slotmap)]
    for s0 in (0.,.99t[end,1])
        println("SMOKE slot=",slot," initial_sg=",s0);flush(stdout)
        r=smoke(t,s0);r["slot"]=slot;push!(results,r)
        open(joinpath(out,"progress.toml"),"w") do io;TOML.print(io,Dict("records"=>results));end
    end
end
open(joinpath(out,"PASS.toml"),"w") do io
    TOML.print(io,Dict("passed"=>true,"constitutive_tables"=>522,"smoke_simulations"=>length(results),
        "records"=>results,"table_sha256"=>bytes2hex(open(sha256,input)),
        "constitutive_receipt_sha256"=>bytes2hex(open(sha256,joinpath(qualified,"constitutive_pass.toml"))),
        "full_fault_receipt_sha256"=>bytes2hex(open(sha256,joinpath(qualified,"full_fault_contract_pass.toml"))),
        "source_sha256"=>bytes2hex(open(sha256,@__FILE__)),
        "full_reservoir_simulation_run"=>false,"physical_closure_validated_against_truth"=>false,
        "production_modified"=>false))
end
