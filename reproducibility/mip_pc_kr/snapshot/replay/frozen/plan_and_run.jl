include("vendor/pipeline/FaultCoreUpscaling.jl")
using .FaultCoreUpscaling, SHA, TOML
const P=FaultCoreUpscaling.Pc

function dictionary(input)
    model,_=P.prepare_model(input)
    @assert all(x->isfinite(x)&&x>=0,input.dictionary)
    pmin=minimum(model.entry);pmax=max(maximum(model.endpoint),maximum(input.dictionary;init=0.))
    @assert 0<pmin<pmax
    coarse=sort!(unique(vcat(0.,.98pmin,10.0.^range(log10(pmin),log10(pmax);length=65),input.dictionary,pmax)))
    # Same pressure-only construction family as the fixed100 study; no MIP/flow
    # results guide these subdivisions, and no extra saturation states are solved.
    pool=coarse
    for round in 1:2
        v=pool[pool.>0];pool=sort!(unique(vcat(pool,sqrt.(v[1:end-1]).*sqrt.(v[2:end]))))
    end
    @assert pool[1]==0 && all(diff(pool).>0)
    pp=P.fixed_from_dictionary(pool,100)
    @assert length(pp)==100&&pp[end]>=maximum(model.endpoint)
    pp,pool
end

function write_raw(path,x,pp,pool)
    @assert !ispath(path)
    open(path,"w") do io
        n=length(x.phi)
        write(io,Int64[2026090904,x.core,n,size(x.perm,2),size(x.edges,1),length(x.inlet),length(x.outlet),
            size(x.tables[1],1),size(x.tables[2],1),length(pp),length(pool),15])
        write(io,Float64[getfield(x.settings,f) for f in fieldnames(P.Settings)])
        for v in (x.perm,x.phi,x.volume,x.edges,x.inlet,x.outlet,x.material,x.tables[1],x.tables[2],pp,pool);write(io,v);end
    end
end

function plan(raw,geometry,out)
    @assert !ispath(out);mkpath(out)
    x=P.read_input(raw);pp,pool=dictionary(x);write_raw(joinpath(out,"raw.bin"),x,pp,pool)
    g=read_geometry(geometry);h=open(sha256,joinpath(out,"raw.bin"))
    g=Geometry(g.core,g.axis,g.length,g.area,h,g.net)
    xx=P.read_input(joinpath(out,"raw.bin"));@assert validate_geometry(xx,g,h)
    write_geometry(joinpath(out,"geometry.bin"),g)
    open(joinpath(out,"sampling.toml"),"w") do io
        TOML.print(io,Dict("policy"=>"fixed100_nearest_log_pressure_only_dictionary_v1", "points"=>100,
            "dictionary_count"=>length(pool),"dictionary_midpoint_rounds"=>2,
            "pressure_max_pa"=>pp[end],"extra_phase_solves_for_dictionary"=>0,
            "interpolation_error_measured"=>false,"continuous_error_bound"=>false))
    end
end

if abspath(PROGRAM_FILE)==@__FILE__
    @assert length(ARGS)==5
    raw,geometry,lib,out,archive=ARGS
    @assert archive in ("true","false")
    plan(raw,geometry,joinpath(out,"planned"))
    run_files(joinpath(out,"planned/raw.bin"),joinpath(out,"planned/geometry.bin"),lib,
        joinpath(out,"coupled");save_states=archive=="true")
end
