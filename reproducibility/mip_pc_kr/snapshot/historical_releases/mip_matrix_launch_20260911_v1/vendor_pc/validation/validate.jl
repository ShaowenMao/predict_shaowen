include(joinpath(@__DIR__,"..","src","FaultCorePc.jl"))
include("json_report.jl")
using .FaultCorePc, SHA
using .FaultCorePc.StandaloneMIP: read_model, readvec, read_state
include("event_audit.jl")

function save_json(path,data)
    @assert !ispath(path)
    open(path,"w") do io;json_value(io,data);println(io);end
end
relative(a,b)=maximum(abs.(a.-b)./max.(abs.(a),abs.(b),floatmin(Float64));init=0.)
bitsame(a,b)=reinterpret(UInt64,a)==reinterpret(UInt64,b)

function validate(input_dir,out;archive=true,same_model_replay=false)
    @assert !ispath(out);mkpath(out)
    raw=joinpath(input_dir,"raw.bin");input=read_input(raw)
    # Compile preparation independently of its reported numerical timer.
    prepare_model(input);t=time_ns();m,audit=prepare_model(input);prep_seconds=(time_ns()-t)/1e9
    refmodel=read_model(joinpath(input_dir,"model.bin"))
    differences=Dict{String,Any}();model_bits=true
    for name in (:scale,:entry,:endpoint,:pv)
        a=getfield(m,name);b=getfield(refmodel,name);d=relative(a,b)
        @assert d<=2e-14 "Local Pc preparation differs beyond the predeclared roundoff gate"
        differences[string(name)]=Dict("relative_error"=>d,"bitwise_equal"=>bitsame(a,b))
        model_bits&=bitsame(a,b)
    end
    for name in (:material,:inlet,:outlet,:row,:neighbor,:pressures)
        @assert getfield(m,name)==getfield(refmodel,name)
    end
    @assert length(m.events)==length(refmodel.events) "Entry/closure event grouping changed"
    @assert relative(m.events,refmodel.events)<=2e-14
    event_identity_gate(m,refmodel)
    if !isempty(input.dictionary)
        @assert fixed_from_dictionary(input.dictionary,length(m.pressures))==m.pressures "Fixed sampling selection changed"
    end
    save_json(joinpath(out,"preparation.json"),Dict("differences"=>differences,"bitwise_model"=>model_bits,
        "audit"=>audit,"pure_julia_preparation_seconds"=>prep_seconds,"sampling_preserved"=>true))
    stats=open(joinpath(input_dir,"reference_stats.bin"),"r") do io
        nr,nc=readvec(io,Int64,2);@assert nr==length(m.pressures)&&nc==8
        data=reshape(readvec(io,Float64,nr*nc),nr,nc);@assert eof(io);data
    end
    refpath=strip(read(joinpath(input_dir,"reference_path.txt"),String))
    prepare_paths(m);t=time_ns();paths=prepare_paths(m);path_seconds=(time_ns()-t)/1e9
    refpaths=prepare_paths(refmodel)
    replay=initial(m.n);replay_workspace=Workspace(m.n)
    previous_ref=initial(m.n);previous_state=initial(m.n)
    w=Workspace(m.n);warm=initial(m.n);advance_pc!(m,warm,0.,w,paths);observe(m,warm,w)
    s=initial(m.n);statebits=true;krbits=true;maxsg=0.;maxkr=0.;maxbulk=0.;maxbp=0.
    times=Float64[];counts=Int[];rows=NamedTuple[];event_audits=Dict{String,Any}[];verified_points=0
    hash=open(sha256,raw);sink=archive ? StateArchive(joinpath(out,"states.bin"),m.n,length(m.pressures),hash) : nothing
    try
        open(refpath,"r") do io
            @assert read(io,Int64)==length(m.pressures)
            for (j,pc) in enumerate(m.pressures)
                previous=s.pressure
                copyto!(previous_state.sg,s.sg);copyto!(previous_state.trapped,s.trapped);copyto!(previous_state.invaded,s.invaded)
                t=time_ns();ne=advance_pc!(m,s,pc,w,paths);row=observe(m,s,w)
                push!(times,(time_ns()-t)/1e9);push!(counts,ne);push!(rows,row)
                ref=read_state(io,m.n);kw=readvec(io,Float64,m.n);kg=readvec(io,Float64,m.n)
                ds=maximum(abs.(s.sg.-ref.sg));dk=max(maximum(abs.(w.kw.-kw)),maximum(abs.(w.kg.-kg)))
                maxsg=max(maxsg,ds);maxkr=max(maxkr,dk)
                @assert s.pressure==ref.pressure && s.invaded==ref.invaded && s.trapped==ref.trapped "MIP topology differs"
                @assert isnan(s.breakthrough)==isnan(ref.breakthrough) "Breakthrough classification differs"
                if isfinite(s.breakthrough)
                    bp=relative([s.breakthrough],[ref.breakthrough]);maxbp=max(maxbp,bp)
                    @assert bp<=2e-14 "Breakthrough pressure differs"
                end
                @assert ds<=2e-12 && dk<=1e-11 "State/local Kr discrepancy exceeds roundoff gates"
                @assert (w.kw.==0)==(kw.==0) && (w.kg.==0)==(kg.==0) "Zero-mobility classification differs"
                @assert pc==stats[j,1] "Prescribed output pressure changed"
                event_report=event_schedule_gate(m,refmodel,previous,pc,ne,Int(stats[j,7]),
                    previous_state,previous_ref,s,ref,paths,refpaths)
                append!(event_audits,event_report)
                if !isempty(event_report)
                    println("AUDITED inactive endpoint-bin difference at sample ",j,"; native schedules and physical states verified")
                end
                if same_model_replay
                    rn=advance_pc!(refmodel,replay,pc,replay_workspace,refpaths)
                    phase!(refmodel,replay,replay_workspace)
                    @assert rn==Int(stats[j,7]) "Same-model event count differs"
                    @assert StandaloneMIP.same_state(replay,ref) "Same-model replay is not bitwise identical"
                    @assert bitsame(replay_workspace.kw,kw)&&bitsame(replay_workspace.kg,kg) "Same-model local Kr differs"
                end
                physical=[row.sg_physical,row.trapped_pv_fraction,row.immobile_pv_fraction,
                          Float64(row.gas_breakthrough),row.invaded_pv_fraction]
                db=maximum(abs.(physical.-stats[j,2:6]));maxbulk=max(maxbulk,db)
                @assert db<=2e-12 "Physical bulk saturation/PV diagnostics differ"
                statebits&=bitsame(s.sg,ref.sg)&&isequal(s.breakthrough,ref.breakthrough)
                krbits&=bitsame(w.kw,kw)&&bitsame(w.kg,kg)
                sink===nothing||write_state!(sink,j,s,w)
                previous_ref=ref;verified_points=j
                if j%10==0
                    println("JULIA PC ",input.core," ",j,"/",length(m.pressures)," events",ne," Sg error ",ds," flags exact")
                    flush(stdout)
                end
            end
            @assert eof(io)
        end
    catch err
        save_json(joinpath(out,"mismatch.json"),Dict("error"=>sprint(showerror,err),"completed_points"=>verified_points,
            "attempted_point"=>length(rows),"inactive_endpoint_audits"=>event_audits,
            "pressure_pa"=>s.pressure,"max_abs_sg"=>maxsg,"max_abs_local_kr"=>maxkr,"production_promotion"=>false))
        rethrow()
    finally
        sink===nothing||close(sink)
    end
    if archive
        seen=Ref(0)
        read_archive(joinpath(out,"states.bin");expected_model_hash=hash) do j,restored,kw,kg
            @assert j==seen[]+1 && restored.pressure==m.pressures[j]
            phase!(m,restored,w)
            @assert bitsame(kw,w.kw)&&bitsame(kg,w.kg)
            seen[]+=1
        end
        @assert seen[]==length(m.pressures)
    end
    open(joinpath(out,"pc_curve.csv"),"w") do io
        println(io,"pressure_pa,sg_physical,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction")
        for row in rows;println(io,join([row.pressure_pa,row.sg_physical,row.trapped_pv_fraction,
             row.immobile_pv_fraction,Int(row.gas_breakthrough),row.invaded_pv_fraction],','));end
    end
    result=Dict("core"=>input.core,"points"=>length(rows),"events"=>sum(counts),"fast_path"=>paths!==nothing,
        "event_validation_policy"=>"exact-native-schedules-with-proved-inactive-one-ulp-endpoint-audit-v1",
        "cross_model_event_bins_exact"=>isempty(event_audits),"inactive_endpoint_audits"=>event_audits,
        "same_model_replay_bitwise_passed"=>same_model_replay,
        "all_topology_and_zero_mobility_exact"=>true,"state_bitwise_equal"=>statebits,"local_kr_bitwise_equal"=>krbits,
        "max_absolute_sg_difference"=>maxsg,"max_absolute_local_kr_difference"=>maxkr,
        "max_absolute_bulk_difference"=>maxbulk,"max_relative_breakthrough_difference"=>maxbp,
        "preparation_seconds"=>prep_seconds,"path_preparation_seconds"=>path_seconds,
        "mip_and_observation_seconds"=>sum(times),"sample_seconds"=>times,
        "breakthrough_pa"=>(isfinite(s.breakthrough) ? s.breakthrough : "not_reached"),"all_equivalence_gates_passed"=>true,
        "native_archive_roundtrip_passed"=>archive,"production_promotion"=>false,"reservoir_export_approved"=>false)
    save_json(joinpath(out,"comparison.json"),result)
    result
end

function main()
    mode,input,out=ARGS
    if mode=="controls"
        @assert !ispath(out);mkpath(out);reports=[]
        for folder in sort(filter(isdir,readdir(input;join=true)))
            push!(reports,validate(folder,joinpath(out,basename(folder));archive=false))
        end
        @assert length(reports)==24
        save_json(joinpath(out,"controls.json"),Dict("controls"=>reports,"all_passed"=>true))
        println("PASS 24 controls / 2400 full-state and local-Kr comparisons")
    else
        @assert mode in ("core","core_recovery")
        r=validate(input,out;same_model_replay=mode=="core_recovery");println("COMPLETE Julia Pc core",r["core"]," max Sg error ",r["max_absolute_sg_difference"])
    end
end
abspath(PROGRAM_FILE)==(@__FILE__) && main()
