module FaultCoreUpscaling
using SHA,TOML,LinearAlgebra
include("Diagnostics.jl")
include("vendor/pc/src/FaultCorePc.jl")
include("vendor/kr/PhaseFlow.jl")
include("vendor/kr/json_report.jl")
const Pc=FaultCorePc
const Flow=PhaseFlow
export Geometry,read_geometry,write_geometry,validate_geometry,run_files

struct Geometry
    core::Int
    axis::Int
    length::Float64
    area::Float64
    raw_hash::Vector{UInt8}
    net::Flow.Net
end

function read_geometry(path)
    open(path,"r") do io
        magic,core,axis,n,ne,ni,no=Flow.readvec(io,Int64,7)
        @assert magic==2026090906 && axis==3 && n>0 && ne>=0 && ni>0 && no>0
        len,area=Flow.readvec(io,Float64,2);hash=Flow.readvec(io,UInt8,32)
        e=reshape(Flow.readvec(io,Int64,2ne),ne,2)
        h1=Flow.readvec(io,Float64,ne);h2=Flow.readvec(io,Float64,ne)
        ci=Flow.readvec(io,Int64,ni);co=Flow.readvec(io,Int64,no)
        hi=Flow.readvec(io,Float64,ni);ho=Flow.readvec(io,Float64,no)
        @assert eof(io)
        Geometry(core,axis,len,area,hash,Flow.Net(n,e,h1,h2,ci,co,hi,ho))
    end
end

function write_geometry(path,g)
    @assert !ispath(path);net=g.net
    open(path,"w") do io
        write(io,Int64[2026090906,g.core,g.axis,net.n,size(net.e,1),length(net.ci),length(net.co)])
        write(io,Float64[g.length,g.area]);write(io,g.raw_hash)
        for a in (net.e,net.h1,net.h2,net.ci,net.co,net.hi,net.ho);write(io,a);end
    end
end

function validate_geometry(input,g,hash)
    net=g.net;n=length(input.phi)
    @assert g.core==input.core && g.axis==3 && g.raw_hash==hash
    @assert net.n==n && net.e==input.edges "MIP and flow grid ordering differ"
    @assert sort(net.ci)==sort(input.inlet) && sort(net.co)==sort(input.outlet)
    @assert isfinite(g.length)&&g.length>0&&isfinite(g.area)&&g.area>0
    @assert length(net.h1)==length(net.h2)==size(net.e,1)
    @assert all(1 .<= net.ci .<= n) && all(1 .<= net.co .<= n)
    @assert all(x->isfinite(x)&&x>0,vcat(net.h1,net.h2,net.hi,net.ho))
    true
end

filehash(path)=Diagnostics.measure(()->open(sha256,path),"file_hash")
bitsame(a,b)=reinterpret(UInt64,a)==reinterpret(UInt64,b)
source_hash()=Diagnostics.source(_source_hash_impl)
function _source_hash_impl()
    paths=String[]
    for (root,dirs,files) in walkdir(@__DIR__)
        filter!(!=("__pycache__"),dirs)
        for name in files
            p=joinpath(root,name)
            p==joinpath(@__DIR__,"source_lock.json")||push!(paths,p)
        end
    end
    bytes2hex(sha256(join([replace(relpath(p,@__DIR__),'\\'=>'/')*":"*bytes2hex(filehash(p)) for p in sort(paths)],"\n")))
end

function run_files(raw,geometry,library,out;save_states=false,cache_states=true,phase_solver=nothing)
    @assert !ispath(out) "Output already exists; refusing overwrite"
    started=time_ns();hash=filehash(raw);gh=filehash(geometry);lh=filehash(library);sh=source_hash()
    input,g=Diagnostics.measure("input_decode") do
        input=Pc.read_input(raw);g=read_geometry(geometry);validate_geometry(input,g,hash);(input,g)
    end
    BLAS.set_num_threads(2);Flow.bridge!(library)
    t=time_ns();m,model_audit=Pc.prepare_model(input);paths=Pc.prepare_paths(m);preparation=(time_ns()-t)/1e9
    @assert length(m.pressures)==100 "Candidate policy is fixed100"
    mkpath(out);rows=NamedTuple[];eventcounts=Int[];owners=Int[]
    phase_seconds=0.;mip_seconds=0.;unique_states=0;fallbacks=0;absolute_seconds=0.
    cache=Dict{String,Any}();state=Pc.initial(m.n);w=Pc.Workspace(m.n)
    archive=save_states ? Pc.StateArchive(joinpath(out,"states.bin"),m.n,length(m.pressures),hash) : nothing
    function emit(io,row)
        Diagnostics.measure("flow_log") do
            json_value(io,row);println(io);flush(io)
        end
    end
    try
        open(joinpath(out,"flow_calls.jsonl"),"w") do log
            function solve(kr,sample,label)
                t=time_ns();q,info=(phase_solver===nothing || label=="intrinsic") ? Flow.conductance(g.net,kr) : phase_solver(g.net,kr,label);seconds=(time_ns()-t)/1e9
                @assert isfinite(q)&&q>=0
                emit(log,Dict("sample"=>sample,"phase"=>label,"q"=>q,"seconds"=>seconds,"info"=>info))
                q,seconds,info["fallback_used"]
            end
            qa,absolute_seconds,af=solve(ones(m.n),0,"intrinsic")
            @assert qa>0 "No spanning intrinsic-conductance path"
            for (j,p) in enumerate(m.pressures)
                t=time_ns();ne=Pc.advance_pc!(m,state,p,w,paths);obs=Pc.observe(m,state,w)
                mip_seconds+=(time_ns()-t)/1e9;push!(eventcounts,ne)
                key=cache_states ? bytes2hex(sha256(vcat(reinterpret(UInt8,w.kw),reinterpret(UInt8,w.kg)))) : ""
                if cache_states && haskey(cache,key)
                    saved=cache[key]
                    @assert bitsame(w.kw,saved.kw)&&bitsame(w.kg,saved.kg) "Cache digest collision"
                    qw,qg=saved.qw,saved.qg;owner=saved.sample
                else
                    qw,tw,fw=solve(w.kw,j,"brine");qg,tg,fg=solve(w.kg,j,"gas")
                    phase_seconds+=tw+tg;fallbacks+=Int(fw)+Int(fg);unique_states+=1;owner=j
                    if cache_states
                        cache[key]=(;kw=copy(w.kw),kg=copy(w.kg),qw,qg,sample=j)
                    end
                end
                push!(owners,owner)
                krw=qw/qa;krg=qg/qa
                @assert 0<=krw<=1+1e-5 && 0<=krg<=1+1e-5
                row=(;pressure_pa=p,sg_physical=obs.sg_physical,krw,krg,
                    trapped_pv_fraction=obs.trapped_pv_fraction,immobile_pv_fraction=obs.immobile_pv_fraction,
                    gas_breakthrough=Int(obs.gas_breakthrough),invaded_pv_fraction=obs.invaded_pv_fraction)
                push!(rows,row);archive===nothing||Pc.write_state!(archive,j,state,w)
                Diagnostics.measure("progress_log") do
                    println("COUPLED core",input.core," sample",j,"/100 Sg ",row.sg_physical," krw ",krw," krg ",krg," owner ",owner);flush(stdout)
                end
            end
            open(joinpath(out,"pc_kr_raw.csv"),"w") do io
                println(io,join(string.(keys(rows[1])),','))
                for r in rows;println(io,join(values(r),','));end
            end
            summary=Dict("core"=>input.core,"axis"=>3,"points"=>length(rows),"unique_mobility_states"=>unique_states,
                "owner_samples"=>owners,"event_counts"=>eventcounts,"events"=>sum(eventcounts),"fast_path"=>paths!==nothing,
                "preparation_seconds"=>preparation,"mip_and_observation_seconds"=>mip_seconds,
                "phase_flow_seconds"=>phase_seconds,"intrinsic_flow_seconds"=>absolute_seconds,
                "phase_fallback_calls"=>fallbacks,"intrinsic_fallback_used"=>af,
                "intrinsic_conductance"=>qa,"intrinsic_permeability_m2"=>qa*g.length/g.area,
                "normalization"=>"fresh Julia intrinsic conductance, same TPFA network and corrected direct solver",
                "model_audit"=>model_audit,"raw_input_sha256"=>bytes2hex(hash),"geometry_sha256"=>bytes2hex(gh),
                "bridge_sha256"=>bytes2hex(lh),"source_tree_sha256"=>sh,"julia_version"=>string(VERSION),
                "blas_threads"=>BLAS.get_num_threads(),"julia_threads"=>Threads.nthreads(),
                "archive_saved"=>save_states,"cache_exact_mobility_states"=>cache_states,
                "production_promotion"=>false,"reservoir_export_approved"=>false)
            Diagnostics.measure("summary_write") do
                open(joinpath(out,"summary.toml"),"w") do io;TOML.print(io,summary);end
            end
        end
    catch err
        open(joinpath(out,"failure.toml"),"w") do io
            TOML.print(io,Dict("error"=>sprint(showerror,err),"completed_points"=>length(rows),"production_promotion"=>false))
        end
        rethrow()
    finally
        archive===nothing||close(archive)
    end
    @assert filehash(raw)==hash && filehash(geometry)==gh && filehash(library)==lh && source_hash()==sh
    outputs=Dict(p=>bytes2hex(filehash(joinpath(out,p))) for p in readdir(out))
    open(joinpath(out,"completion.toml"),"w") do io
        TOML.print(io,Dict("execution_complete"=>true,"outputs"=>outputs,"core"=>input.core,"points"=>100,
            "wall_seconds"=>(time_ns()-started)/1e9,"numerical_workflow"=>"Julia Pc/MIP and corrected-direct phase flow",
            "validation_passed"=>false,"production_promotion"=>false,"reservoir_export_approved"=>false))
    end
    rows
end
end
