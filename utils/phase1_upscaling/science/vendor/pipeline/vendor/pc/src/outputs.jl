function observe(m,s,w)
    phase!(m,s,w);pv=sum(m.pv)
    (pressure_pa=s.pressure,sg_physical=sum(m.pv.*s.sg)/pv,
     trapped_pv_fraction=sum(m.pv[s.trapped.!=0])/pv,
     immobile_pv_fraction=sum(m.pv[w.kw.==0])/pv,
     gas_breakthrough=isfinite(s.breakthrough),invaded_pv_fraction=sum(m.pv[s.invaded.!=0])/pv)
end

function run_file(raw_path,out;backend=:auto)
    @assert !ispath(out) "Output directory must be new; existing results are never overwritten"
    input=read_input(raw_path);m,audit=prepare_model(input)
    if !isempty(input.dictionary)
        @assert fixed_from_dictionary(input.dictionary,length(input.pressures))==input.pressures
    end
    hash=open(sha256,raw_path);mkpath(out)
    archive=StateArchive(joinpath(out,"states.bin"),m.n,length(m.pressures),hash)
    csv=open(joinpath(out,"pc_curve.csv"),"w");began=time_ns()
    println(csv,"pressure_pa,sg_physical,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction")
    result=try
        upscale(m;backend=backend,callback=(j,s,w,row)->begin
            write_state!(archive,j,s,w)
            println(csv,join([row.pressure_pa,row.sg_physical,row.trapped_pv_fraction,
                row.immobile_pv_fraction,Int(row.gas_breakthrough),row.invaded_pv_fraction],','))
            flush(csv)
            println("PC sample ",j,"/",length(m.pressures)," pressure ",s.pressure," Pa")
        end)
    finally
        close(archive);close(csv)
    end
    @assert open(sha256,raw_path)==hash
    source_files=[joinpath(@__DIR__,name) for name in ("FaultCorePc.jl","paths.jl","outputs.jl")]
    push!(source_files,joinpath(@__DIR__,"..","vendor","StandaloneMIP.jl"))
    metadata=Dict("schema"=>"FaultCorePc.native.v1","core"=>input.core,"points"=>length(result.rows),
        "julia_version"=>string(VERSION),"input_sha256"=>bytes2hex(hash),"cpu"=>Sys.CPU_NAME,
        "state_archive_sha256"=>bytes2hex(open(sha256,joinpath(out,"states.bin"))),
        "curve_sha256"=>bytes2hex(open(sha256,joinpath(out,"pc_curve.csv"))),
        "source_files"=>Dict(basename(p)=>bytes2hex(open(sha256,p)) for p in source_files),
        "settings"=>Dict(string(k)=>getfield(input.settings,k) for k in fieldnames(Settings)),
        "execution_complete"=>true,"reservoir_export_approved"=>false,"production_promotion"=>false,
        "fast_path"=>result.fast_path,"events"=>sum(result.event_counts),
        "breakthrough_reached"=>isfinite(result.breakthrough_pa),"audit"=>audit,
        "run_seconds_including_outputs"=>(time_ns()-began)/1e9,
        "saturation_convention"=>"physical PV average, including pre-breakthrough gas; no synthetic reservoir interpolation")
    if isfinite(result.breakthrough_pa);metadata["breakthrough_pa"]=result.breakthrough_pa;end
    open(joinpath(out,"completion.toml"),"w") do io;TOML.print(io,metadata;sorted=true);end
    result
end

function upscale(m;backend=:auto,callback=(j,s,w,row)->nothing)
    paths=prepare_paths(m;backend=backend);w=Workspace(m.n);s=initial(m.n)
    rows=NamedTuple[];events=Int[]
    for (j,pc) in enumerate(m.pressures)
        push!(events,advance_pc!(m,s,pc,w,paths));row=observe(m,s,w)
        push!(rows,row);callback(j,s,w,row)
    end
    (rows=rows,final_state=s,event_counts=events,fast_path=paths!==nothing,
     breakthrough_pa=s.breakthrough,reservoir_export_approved=false)
end

struct StateArchive
    io::IOStream
    n::Int
    points::Int
    model_hash::Vector{UInt8}
end

function StateArchive(path::AbstractString,n::Int,points::Int,model_hash::Vector{UInt8})
    @assert !ispath(path) && n>0 && points>0 && length(model_hash)==32
    io=open(path,"w")
    write(io,Int64[2026090905,n,points]);write(io,model_hash)
    StateArchive(io,n,points,model_hash)
end
Base.close(a::StateArchive)=close(a.io)
function write_state!(a::StateArchive,j,s,w)
    @assert 1<=j<=a.points && length(s.sg)==a.n
    # Fixed-size records make partial archives detectable and permit streaming Kr input.
    expected=24+32+(j-1)*(24+26a.n)
    @assert position(a.io)==expected "Out-of-order or duplicate state write"
    write(a.io,Int64(j));write(a.io,[s.pressure,s.breakthrough]);write(a.io,s.sg)
    write(a.io,s.invaded);write(a.io,s.trapped);write(a.io,w.kw);write(a.io,w.kg)
    flush(a.io)
end

read_archive(callback::Function,path::AbstractString;kwargs...)=read_archive(path,callback;kwargs...)
function read_archive(path::AbstractString,callback;expected_model_hash=nothing)
    open(path,"r") do io
        magic,n,points=readvec(io,Int64,3);@assert magic==2026090905 && n>0 && points>0
        hash=readvec(io,UInt8,32)
        expected_model_hash===nothing||@assert hash==expected_model_hash
        @assert filesize(path)==24+32+points*(24+26n) "Incomplete or trailing archive data"
        for j in 1:points
            @assert read(io,Int64)==j
            s=read_state(io,n);kw=readvec(io,Float64,n);kg=readvec(io,Float64,n)
            callback(j,s,kw,kg)
        end
        @assert eof(io)
    end
end
