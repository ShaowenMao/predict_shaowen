# Same pinned numerical functions as Kernel.jl; only the queried-state schedule changes.
function run_cached_adaptive(input,g,out,identity; qualification_reference=false)
    @assert !ispath(out); mkpath(out)
    start=time_ns()
    t=time_ns(); m,audit=Pc.prepare_model(input); paths=Pc.prepare_paths(m)
    prep=(time_ns()-t)/1e9
    println("MODEL fast_path=",paths!==nothing," cells=",m.n,
            " maximum_entry=",maximum(m.entry)," minimum_endpoint=",minimum(m.endpoint));flush(stdout)
    t=time_ns(); drainage=cached_pc_pass(m,paths,identity); planned=drainage.rows
    planning_seconds=(time_ns()-t)/1e9
    reference_seconds=0.0
    Backend.reset!(); solver=Backend.Solver(U.Flow)
    cache=Dict{String,Any}(); results=Dict{Int,Any}()
    phase_seconds=0.; intrinsic_seconds=0.; replay_seconds=0.; hash_seconds=0.
    calls=0; visited=0
    try
        open(joinpath(out,"flow_calls.jsonl"),"w") do log
            function solve(kr,j,label)
                t=time_ns()
                q,info=label=="intrinsic" ? U.Flow.conductance(g.net,kr) : solver(g.net,kr,label)
                dt=(time_ns()-t)/1e9
                @assert isfinite(q) && q >= 0
                U.json_value(log,Dict("sample"=>j,"phase"=>label,"q"=>q,"seconds"=>dt,"info"=>info))
                println(log); flush(log)
                q,dt
            end
            qa,intrinsic_seconds=solve(ones(m.n),0,"intrinsic")
            @assert qa > 0
            function query(j)
                @assert !haskey(results,j) && length(results)<10
                t=time_ns()
                state,w=restore_drainage(drainage,m,j,identity)
                hash_seconds+=(time_ns()-t)/1e9; visited+=1
                key=bytes2hex(sha256(vcat(reinterpret(UInt8,w.kw),reinterpret(UInt8,w.kg))))
                if haskey(cache,key)
                    saved=cache[key]
                    @assert U.bitsame(w.kw,saved.kw) && U.bitsame(w.kg,saved.kg)
                    qw,qg,owner=saved.qw,saved.qg,saved.sample
                else
                    qw,tw=solve(w.kw,j,"brine"); qg,tg=solve(w.kg,j,"gas")
                    phase_seconds+=tw+tg; calls+=2; owner=j
                    cache[key]=(;kw=copy(w.kw),kg=copy(w.kg),qw,qg,sample=j)
                end
                @assert 0 <= qw/qa <= 1+1e-5 && 0 <= qg/qa <= 1+1e-5
                results[j]=(sample=j,pressure_pa=m.pressures[j],sg_physical=planned[j].sg_physical,
                            krw=qw/qa,krg=qg/qa,owner=owner)
                println("QUERY adaptive core=",input.core," sample=",j," states=",length(results)); flush(stdout)
                [qw/qa,qg/qa]
            end
            t=time_ns()
            chosen,order=AdaptiveSelector.select_adaptive([r.sg_physical for r in planned],query)
            decision_plus_queries=(time_ns()-t)/1e9
            flowrows=[(sample=j,pressure_pa=results[j].pressure_pa,sg_physical=results[j].sg_physical,
                       krw=results[j].krw,krg=results[j].krg) for j in chosen]
            writecsv(joinpath(out,"pc100.csv"),planned)
            writecsv(joinpath(out,"kr.csv"),flowrows)
            writecsv(joinpath(out,"selection.csv"),[(rank=k,sample=j,query_order=findfirst(==(j),order),
                       actual_sg=planned[j].sg_physical,pressure_pa=planned[j].pressure_pa) for (k,j) in enumerate(chosen)])
            Backend.clear!(solver)
            candidate_wall=(time_ns()-start)/1e9
            if qualification_reference
                reference_seconds=verify_forward_reference(drainage,m,paths,identity)
            end
            summary=Dict("mode"=>"adaptive10","cells"=>m.n,"core"=>input.core,"pc_points"=>100,
                "kr_states"=>length(chosen),"selected_indices"=>chosen,"query_order"=>order,
                "phase_solver_calls"=>calls,"intrinsic_solver_calls"=>1,
                "preparation_seconds"=>prep,"planning_mip_seconds"=>planning_seconds,
                "replay_seconds"=>0.0,"replay_states_checked"=>0,
                "state_access"=>"verified_per_core_cache_v1","cached_states_verified"=>100,
                "cached_queries_verified"=>visited,"cache_payload_bytes"=>drainage.payload_bytes,
                "reference_states_checked"=>(qualification_reference ? 100 : 0),
                "qualification_reference_seconds"=>reference_seconds,"fast_path"=>paths!==nothing,
                "validation_hash_seconds"=>hash_seconds,"phase_flow_seconds"=>phase_seconds,
                "decision_plus_queries_seconds"=>decision_plus_queries,"intrinsic_flow_seconds"=>intrinsic_seconds,
                "wall_seconds"=>candidate_wall,"intrinsic_conductance"=>qa,
                "intrinsic_permeability_m2"=>qa*g.length/g.area,"memory"=>MemoryProbe.snapshot(),
                "native_lifetime"=>Backend.report(),"model_audit"=>audit,
                "same_mip_microstates"=>true,"production_modified"=>false)
            open(joinpath(out,"summary.toml"),"w") do io; TOML.print(io,summary); end
            planned,flowrows
        end
    finally
        Backend.clear!(solver)
    end
end
