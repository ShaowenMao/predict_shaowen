function writecsv(path, rows)
    open(path,"w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows; println(io, join(values(row), ',')); end
    end
end
function digest_state(state, w)
    ctx = SHA.SHA2_256_CTX()
    for a in (state.sg, state.invaded, state.trapped, w.kw, w.kg)
        SHA.update!(ctx, reinterpret(UInt8, a))
    end
    bytes2hex(SHA.digest!(ctx))
end
function pc_pass(m, paths)
    state = Pc.initial(m.n); w = Pc.Workspace(m.n)
    rows = NamedTuple[]
    for (j,p) in enumerate(m.pressures)
        ne = Pc.advance_pc!(m,state,p,w,paths)
        o = Pc.observe(m,state,w)
        push!(rows,(sample=j,pressure_pa=p,sg_physical=o.sg_physical,
            trapped_pv_fraction=o.trapped_pv_fraction,immobile_pv_fraction=o.immobile_pv_fraction,
            gas_breakthrough=Int(o.gas_breakthrough),invaded_pv_fraction=o.invaded_pv_fraction,
            events=ne,state_hash=digest_state(state,w)))
    end
    rows
end

function run_curve(input,g,out,mode)
    @assert !ispath(out); mkpath(out)
    start = time_ns()
    t = time_ns(); m,audit = Pc.prepare_model(input); paths = Pc.prepare_paths(m)
    prep = (time_ns()-t)/1e9
    selection_seconds = 0.0; planning_seconds = 0.0
    planned = nothing
    if mode == "sparse10"
        t = time_ns(); planned = pc_pass(m, paths); planning_seconds = (time_ns()-t)/1e9
        t = time_ns(); chosen,targets = select_states([r.sg_physical for r in planned],10)
        selection_seconds = (time_ns()-t)/1e9
        writecsv(joinpath(out,"selection.csv"),[(rank=k,sample=j,target_sg=targets[k],
            actual_sg=planned[j].sg_physical,pressure_pa=planned[j].pressure_pa) for (k,j) in enumerate(chosen)])
    else
        @assert mode == "full100"
        chosen = collect(eachindex(m.pressures))
    end
    Backend.reset!(); solver = Backend.Solver(U.Flow)
    state = Pc.initial(m.n); w = Pc.Workspace(m.n)
    pcrows = NamedTuple[]; flowrows = NamedTuple[]
    cache = Dict{String,Any}()
    phase_seconds = 0.; intrinsic_seconds = 0.; mip_seconds = 0.; audit_seconds = 0.
    calls = 0; owners = Int[]; memory = MemoryProbe.snapshot()
    try
        open(joinpath(out,"flow_calls.jsonl"),"w") do log
            function solve(kr,j,label)
                t = time_ns()
                q,info = label == "intrinsic" ? U.Flow.conductance(g.net,kr) : solver(g.net,kr,label)
                dt = (time_ns()-t)/1e9
                @assert isfinite(q) && q >= 0
                U.json_value(log,Dict("sample"=>j,"phase"=>label,"q"=>q,"seconds"=>dt,"info"=>info))
                println(log); flush(log)
                q,dt
            end
            qa,intrinsic_seconds = solve(ones(m.n),0,"intrinsic")
            @assert qa > 0
            for (j,p) in enumerate(m.pressures)
                t = time_ns(); ne = Pc.advance_pc!(m,state,p,w,paths); o = Pc.observe(m,state,w)
                mip_seconds += (time_ns()-t)/1e9
                t = time_ns()
                row = (sample=j,pressure_pa=p,sg_physical=o.sg_physical,
                    trapped_pv_fraction=o.trapped_pv_fraction,immobile_pv_fraction=o.immobile_pv_fraction,
                    gas_breakthrough=Int(o.gas_breakthrough),invaded_pv_fraction=o.invaded_pv_fraction,
                    events=ne,state_hash=digest_state(state,w))
                planned === nothing || @assert row == planned[j] "MIP replay changed a saved microstate"
                push!(pcrows,row); audit_seconds += (time_ns()-t)/1e9
                j in chosen || continue
                key = bytes2hex(sha256(vcat(reinterpret(UInt8,w.kw),reinterpret(UInt8,w.kg))))
                if haskey(cache,key)
                    saved = cache[key]
                    @assert U.bitsame(w.kw,saved.kw) && U.bitsame(w.kg,saved.kg)
                    qw,qg,owner = saved.qw,saved.qg,saved.sample
                else
                    qw,tw = solve(w.kw,j,"brine"); qg,tg = solve(w.kg,j,"gas")
                    phase_seconds += tw+tg; calls += 2; owner=j
                    cache[key] = (;kw=copy(w.kw),kg=copy(w.kg),qw,qg,sample=j)
                end
                push!(owners,owner)
                @assert 0 <= qw/qa <= 1+1e-5 && 0 <= qg/qa <= 1+1e-5
                push!(flowrows,(sample=j,pressure_pa=p,sg_physical=o.sg_physical,krw=qw/qa,krg=qg/qa))
                println("PROGRESS ",mode," core=",input.core," sample=",j," states=",length(flowrows)); flush(stdout)
            end
            writecsv(joinpath(out,"pc100.csv"),pcrows)
            writecsv(joinpath(out,"kr.csv"),flowrows)
            Backend.clear!(solver)
            memory = MemoryProbe.snapshot()
            summary = Dict("mode"=>mode,"cells"=>m.n,"core"=>input.core,"pc_points"=>length(pcrows),
                "kr_states"=>length(flowrows),"selected_indices"=>chosen,"phase_solver_calls"=>calls,
                "intrinsic_solver_calls"=>1,"owner_samples"=>owners,"fast_path"=>paths!==nothing,
                "preparation_seconds"=>prep,"planning_mip_seconds"=>planning_seconds,
                "selection_seconds"=>selection_seconds,"mip_observation_seconds"=>mip_seconds,
                "phase_flow_seconds"=>phase_seconds,"intrinsic_flow_seconds"=>intrinsic_seconds,
                "validation_hash_seconds"=>audit_seconds,"wall_seconds"=>(time_ns()-start)/1e9,
                "intrinsic_conductance"=>qa,"intrinsic_permeability_m2"=>qa*g.length/g.area,
                "memory"=>memory,"native_lifetime"=>Backend.report(),"model_audit"=>audit,
                "same_mip_microstates"=>true,"production_modified"=>false)
            open(joinpath(out,"summary.toml"),"w") do io; TOML.print(io,summary); end
        end
    finally
        Backend.clear!(solver)
    end
    pcrows,flowrows
end
