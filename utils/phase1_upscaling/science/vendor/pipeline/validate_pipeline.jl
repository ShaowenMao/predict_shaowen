include("FaultCoreUpscaling.jl")
using .FaultCoreUpscaling,TOML,SHA
const U=FaultCoreUpscaling
const Pc=U.Pc
include("vendor/pc/validation/event_audit.jl")
function main(raw,geometry_dir,reference_dir,out,receipt)
@assert !ispath(receipt)
csv(path)=reduce(vcat,[permutedims(parse.(Float64,split(x,','))) for x in readlines(path)])
completion=TOML.parsefile(joinpath(out,"completion.toml"));@assert completion["execution_complete"]
for (name,hash) in completion["outputs"]
    @assert bytes2hex(open(sha256,joinpath(out,name)))==hash
end
summary=TOML.parsefile(joinpath(out,"summary.toml"))
lines=readlines(joinpath(out,"pc_kr_raw.csv"))
got=reduce(vcat,[permutedims(parse.(Float64,split(x,','))) for x in lines[2:end]])
expected=csv(joinpath(geometry_dir,"expected.csv"))
@assert size(got)==size(expected)==(100,8)
@assert got[:,1]==expected[:,1]
bulk=maximum(abs.(got[:,[2,5,6,7,8]].-expected[:,[2,5,6,7,8]]));@assert bulk<=2e-12
@assert (got[:,3:4].==0)==(expected[:,3:4].==0) "Effective phase zero-flow classification differs"
rel=abs.(got[:,3:4].-expected[:,3:4])./max.(abs.(got[:,3:4]),abs.(expected[:,3:4]),floatmin(Float64))
maxrel=maximum(rel);maxabs=maximum(abs.(got[:,3:4].-expected[:,3:4]))
@assert maxrel<=1e-5 "Coupled effective Kr mismatch"
qref=parse(Float64,strip(read(joinpath(geometry_dir,"qa_reference.txt"),String)))
qr=abs(summary["intrinsic_conductance"]-qref)/max(abs(summary["intrinsic_conductance"]),abs(qref))
@assert qr<=1e-5
input=Pc.read_input(raw);m,_=Pc.prepare_model(input)
ref=Pc.StandaloneMIP.read_model(joinpath(reference_dir,"model.bin"))
for name in (:scale,:entry,:endpoint,:pv)
    a=getfield(m,name);b=getfield(ref,name)
    @assert maximum(abs.(a.-b)./max.(abs.(a),abs.(b),floatmin(Float64)))<=2e-14
end
@assert m.pressures==ref.pressures && m.material==ref.material
event_identity_gate(m,ref);paths=Pc.prepare_paths(m);refpaths=Pc.prepare_paths(ref)
stats=open(joinpath(reference_dir,"reference_stats.bin")) do io
    nr,nc=Pc.StandaloneMIP.readvec(io,Int64,2)
    reshape(Pc.StandaloneMIP.readvec(io,Float64,nr*nc),nr,nc)
end
old=Pc.initial(m.n);oldref=Pc.initial(m.n);workspace=Pc.Workspace(m.n)
maxsg=0.;maxlocal=0.;maxbp=0.;audits=Dict{String,Any}[];cache=Dict{Int,Any}();seen=0
reference_path=strip(read(joinpath(reference_dir,"reference_path.txt"),String))
open(reference_path) do io
    @assert read(io,Int64)==100
    Pc.read_archive(joinpath(out,"states.bin");expected_model_hash=open(sha256,raw)) do j,s,kw,kg
        r=Pc.StandaloneMIP.read_state(io,m.n)
        rw=Pc.StandaloneMIP.readvec(io,Float64,m.n);rg=Pc.StandaloneMIP.readvec(io,Float64,m.n)
        @assert j==seen+1 && s.pressure==r.pressure==got[j,1]
        @assert s.invaded==r.invaded && s.trapped==r.trapped
        @assert (kw.==0)==(rw.==0) && (kg.==0)==(rg.==0)
        @assert isnan(s.breakthrough)==isnan(r.breakthrough)
        if isfinite(s.breakthrough)
            maxbp=max(maxbp,abs(s.breakthrough-r.breakthrough)/max(abs(s.breakthrough),abs(r.breakthrough)))
        end
        @assert maxbp<=2e-14
        ds=maximum(abs.(s.sg.-r.sg));dk=max(maximum(abs.(kw.-rw)),maximum(abs.(kg.-rg)))
        maxsg=max(maxsg,ds);maxlocal=max(maxlocal,dk)
        @assert ds<=2e-12 && dk<=1e-11
        Pc.phase!(m,s,workspace)
        @assert U.bitsame(kw,workspace.kw)&&U.bitsame(kg,workspace.kg) "Flow handoff mobility differs from Julia MIP state"
        obs=Pc.observe(m,s,workspace)
        @assert abs(obs.sg_physical-got[j,2])<=2e-12
        append!(audits,event_schedule_gate(m,ref,old.pressure,s.pressure,summary["event_counts"][j],
            Int(stats[j,7]),old,oldref,s,r,paths,refpaths))
        owner=summary["owner_samples"][j];@assert 1<=owner<=j
        if owner==j;cache[j]=(kw,kg)
        else;@assert U.bitsame(kw,cache[owner][1])&&U.bitsame(kg,cache[owner][2]);end
        old=s;oldref=r;seen=j
    end
    @assert eof(io)&&seen==100
end
result=Dict("core"=>summary["core"],"points"=>100,"all_coupled_checks_passed"=>true,
    "max_absolute_kr_difference"=>maxabs,"max_relative_kr_difference"=>maxrel,
    "intrinsic_conductance_relative_difference"=>qr,"max_absolute_bulk_difference"=>bulk,
    "max_absolute_local_sg_difference"=>maxsg,"max_absolute_local_kr_difference"=>maxlocal,
    "max_relative_breakthrough_difference"=>maxbp,"topology_and_zero_classifications_exact"=>true,
    "native_archive_handoff_bitwise_verified"=>true,"cache_exactness_verified"=>true,
    "cross_model_event_bins_exact"=>isempty(audits),"inactive_endpoint_audits"=>audits,
    "production_promotion"=>false,"reservoir_export_approved"=>false)
open(receipt,"w") do io;U.json_value(io,result);println(io);end
println("PASS coupled Pc/Kr core",summary["core"]," max Kr relative ",maxrel," max Sg ",maxsg)
end
main(ARGS...)
