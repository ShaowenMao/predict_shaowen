include(joinpath(ENV["FROZEN_SOURCE"], "plan_and_run.jl"))
using DelimitedFiles, Test

function clay60(t)
    @assert size(t)==(32,4) && t[1,1]==0 && t[end,1]==0.70
    # The original WRR generator uses a*exp(b*Sg). Recover the frozen fit,
    # not a new fit, and verify it against every positive-Sg table node.
    b=log(t[end,4]/t[2,4])/(t[end,1]-t[2,1])
    a=t[2,4]/exp(b*t[2,1])
    residual=maximum(abs.(a.*exp.(b.*t[2:end,1])./t[2:end,4].-1))
    @assert residual<1e-11 "Frozen clay table is not the documented WRR exponential"
    ratio=cosd(60.)/cosd(30.)
    sg=vcat(0.,0.001,collect(range(0.,0.8;length=30))[2:end-1],0.799,0.8)
    swstar=1 .- sg./0.8
    result=hcat(sg,(1 .- swstar).^2.5,swstar.^3.5,a*ratio.*exp.(b.*sg))
    result[1,4]=0.
    @assert result[end,2]==1 && result[end,3]==0 && result[end,1]==0.8
    result,Dict("reference_pc_law"=>"a*exp(b*Sg), with the existing Sg=0 export node",
        "reference_a30_pa"=>a,"reference_b"=>b,"reference_a60_pa"=>a*ratio,
        "positive_node_relative_reconstruction_error"=>residual,
        "contact_angle_ratio"=>ratio,"clay_swi"=>0.2,"clay_nw"=>3.5,"clay_ng"=>2.5,
        "reference_nodes"=>32,"reference_node_policy"=>"original WRR generator, endpoint 0.8",
        "source"=>"WRR Appendix C1.1/C2.1 and original relPermAnalysis.m lines 130-166, 567-576",
        "new_curve_fit_performed"=>false)
end

function transform(raw,geometry,out)
    @assert !ispath(out);mkpath(out)
    x=P.read_input(raw);g=read_geometry(geometry)
    @assert validate_geometry(x,g,open(sha256,raw))
    @assert x.settings.co2_brine_theta_deg==30 && x.settings.clay_endpoint==0.70
    @assert length(x.pressures)==100 && x.settings.sand_endpoint==0.95
    t,reference=clay60(x.tables[2])
    fields=fieldnames(P.Settings)
    cfg=P.Settings((f==:co2_brine_theta_deg ? 60. : f==:clay_endpoint ? 0.8 : getfield(x.settings,f) for f in fields)...)
    y=P.CoreInput(x.core,x.perm,x.phi,x.volume,x.edges,x.inlet,x.outlet,x.material,
        (copy(x.tables[1]),t),copy(x.pressures),copy(x.dictionary),cfg)
    m30,_=P.prepare_model(x);m60,audit=P.prepare_model(y)
    sand=findall(==(1),x.material);clay=findall(==(2),x.material)
    @assert m60.entry[sand]==m30.entry[sand] && m60.endpoint[sand]==m30.endpoint[sand]
    @assert m60.pv==m30.pv && m60.pressures==m30.pressures
    @assert maximum(m60.endpoint)<=x.pressures[end] "Existing pressure range does not cover 60-degree endpoints"
    c30=m30.curves[2];c60=m60.curves[2]
    p30=P.StandaloneMIP.interpolate(c30.sg,c30.pc,cfg.clay_anchor_sg)
    p60=P.StandaloneMIP.interpolate(c60.sg,c60.pc,cfg.clay_anchor_sg)
    ratio=cosd(60.)/cosd(30.)
    @assert all(isapprox(m60.scale[i]*p60/(m30.scale[i]*p30),ratio;rtol=1e-13) for i in clay)
    @assert P.StandaloneMIP.interpolate(c60.sg,c60.brine,0.799)>0
    @assert P.StandaloneMIP.interpolate(c60.sg,c60.brine,0.8)==0
    write_raw(joinpath(out,"raw.bin"),y,y.pressures,y.dictionary)
    h=open(sha256,joinpath(out,"raw.bin"))
    write_geometry(joinpath(out,"geometry.bin"),Geometry(g.core,g.axis,g.length,g.area,h,g.net))
    z=P.read_input(joinpath(out,"raw.bin"));gz=read_geometry(joinpath(out,"geometry.bin"))
    @assert validate_geometry(z,gz,h)
    for f in (:perm,:phi,:volume,:edges,:inlet,:outlet,:material,:pressures,:dictionary)
        @assert isequal(getfield(z,f),getfield(x,f))
    end
    @assert z.tables[1]==x.tables[1] && z.tables[2]==t
    @assert all(getfield(z.settings,f)==getfield(x.settings,f) for f in fields if !(f in (:clay_endpoint,:co2_brine_theta_deg)))
    for f in fieldnames(typeof(g.net));@assert isequal(getfield(gz.net,f),getfield(g.net,f));end
    @assert gz.length==g.length && gz.area==g.area && gz.axis==g.axis
    reference["entry_pressure_pa"]=minimum(m60.entry[m60.inlet])
    reference["baseline_entry_pressure_pa"]=minimum(m30.entry[m30.inlet])
    reference["clay_reference_anchor_pa"]=audit["clay_reference_anchor_pa"]
    reference["clay_conversion_factor"]=audit["clay_conversion_factor"]
    reference["maximum_local_endpoint_pa"]=maximum(m60.endpoint)
    reference["maximum_sampled_pressure_pa"]=x.pressures[end]
    reference["unchanged_geometry_intrinsic_properties_sand_and_sampling"]=true
    reference["clay_anchor_scales_once_by_cosine_ratio"]=true
    reference["core"]=x.core
    reference["raw_parent_sha256"]=bytes2hex(open(sha256,raw))
    reference["passed"]=true
    open(joinpath(out,"scenario_audit.toml"),"w") do io;TOML.print(io,reference);end
    writedlm(joinpath(out,"clay_reference_60.csv"),t,',')
    reference
end

function compare_control(raw,geometry,lib,out,baseline)
    old=readdlm(baseline,',';skipstart=1)
    new=readdlm(joinpath(out,"pc_kr_raw.csv"),',';skipstart=1)
    @assert size(old)==size(new)==(100,8)
    before=TOML.parsefile(joinpath(dirname(baseline),"summary.toml"))
    after=TOML.parsefile(joinpath(out,"summary.toml"))
    @assert after["raw_input_sha256"]==bytes2hex(open(sha256,raw))==before["raw_input_sha256"]
    @assert after["geometry_sha256"]==bytes2hex(open(sha256,geometry))==before["geometry_sha256"]
    @assert after["bridge_sha256"]==bytes2hex(open(sha256,lib))
    @assert after["source_tree_sha256"]==FaultCoreUpscaling.source_hash()
    completion=TOML.parsefile(joinpath(out,"completion.toml"))
    @assert completion["execution_complete"]
    for (p,h) in completion["outputs"]
        @assert bytes2hex(open(sha256,joinpath(out,p)))==h
    end
    delta=vec(maximum(abs.(old.-new);dims=1))
    println("CONTROL max absolute differences by column: ",delta)
    @testset "Unchanged 30-degree control" begin
        @test old[:,[1,7]]==new[:,[1,7]]
        # PV reductions can differ in their last bits across compute-node CPUs.
        @test maximum(delta[[2,5,6,8]])<=16eps(1.)
        @test before["event_counts"]==after["event_counts"]
        @test before["owner_samples"]==after["owner_samples"]
        @test maximum(delta[3:4])<1e-8
        @test all(abs.(old[:,3:4].-new[:,3:4]).<=1e-12 .+ 1e-5.*abs.(old[:,3:4]))
    end
    open(joinpath(out,"paired_control_pass.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"max_absolute_difference_by_column"=>delta,
            "pressure_breakthrough_events_and_cache_owners_exact"=>true,
            "pv_aggregate_roundoff_tolerance"=>16eps(1.),
            "max_absolute_kr_difference"=>maximum(delta[3:4])))
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    mode,raw,geometry,lib,out=ARGS[1:5]
    if mode=="transform"
        transform(raw,geometry,out)
    elseif mode=="run"
        transform(raw,geometry,joinpath(out,"planned"))
        run_files(joinpath(out,"planned/raw.bin"),joinpath(out,"planned/geometry.bin"),lib,
            joinpath(out,"coupled");save_states=false)
    elseif mode=="control30"
        run_files(raw,geometry,lib,out;save_states=false)
        compare_control(raw,geometry,lib,out,ARGS[6])
    elseif mode=="validate30"
        compare_control(raw,geometry,lib,out,ARGS[6])
    else
        error("Unknown mode")
    end
end
