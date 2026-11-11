using TOML,SHA,LinearAlgebra,SparseArrays,Serialization,Test
const cfg=TOML.parsefile(ARGS[1])
const SRC=cfg["science"]
include(joinpath(SRC,"vendor/pipeline/FaultCoreUpscaling.jl"))
const U=FaultCoreUpscaling
const Pc=U.Pc
const P=Pc
include(joinpath(SRC,"vendor/kernel/Kernel.jl"))
include(joinpath(SRC,"vendor/PressurePlan.jl"))
include(joinpath(SRC,"ClayPc40.jl"))
include("StrictProbe.jl")
const RUN=TOML.parsefile(joinpath(SRC,"runtime.toml"))
@assert string(VERSION)=="1.12.6" && Threads.nthreads()==2
@assert bytes2hex(open(sha256,RUN["bridge"]))==RUN["bridge_sha256"]
BLAS.set_num_threads(2)
U.Flow.bridge!(RUN["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,RUN["pipeline_hash"])

function jsonfile(path,value)
    @assert !ispath(path)
    open(path,"w") do io
        U.json_value(io,value);println(io);flush(io)
        @assert ccall(:fsync,Cint,(Cint,),fd(io))==0
    end
end

@testset "Unchanged strict gates and analytic operator" begin
    m=[1e-12,1.,1.,1.,1e-12,1e-12,0.,1.]
    @test StrictProbe.reached(m,1e-12,true)
    @test !StrictProbe.reached(m,2e-11,true)
    @test !StrictProbe.reached(m,2e-9,false)
    @test !StrictProbe.reached(m,NaN,false)
    for n in (3,9), contrast in (1.,1e-6)
        e=hcat(collect(1:n-1),collect(2:n))
        net=U.Flow.Net(n,e,fill(2.,n-1),fill(2.,n-1),[1],[n],[2.],[2.])
        k=fill(contrast,n); k[1]=1.
        p,A,_,keep,_=U.Flow.prepare(net,k)
        signature=U.Flow.parts_hash(p,keep)
        d=StrictProbe.two_checks(U.Flow,p,A;deadline=time()+60)
        @test d["passed"]
        @test abs(d["reference_q"]-1/sum(1 ./ k))/(1/sum(1 ./ k))<=1e-9
        @test U.Flow.parts_hash(p,keep)==signature
    end
end
jsonfile(joinpath(cfg["out"],"TEST_PASS.json"),Dict("passed"=>true))

r=cfg["cases"][1]
x=Pc.read_input(r["raw"]);g=U.read_geometry(r["geometry"])
@assert x.core==r["slot"]==g.core && x.settings.co2_brine_theta_deg==40
@assert x.settings.clay_endpoint==.8 && x.settings.sand_endpoint==.95
table=ClayPc40.clay_table(x.tables[2],ClayPc40.reference(SRC,r["sample"],"master"))
@assert maximum(abs.(table.-x.tables[2]))<5e-8
x=Pc.CoreInput(x.core,x.perm,x.phi,x.volume,x.edges,x.inlet,x.outlet,x.material,
              (x.tables[1],table),x.pressures,x.dictionary,x.settings)
pp,pool=dictionary(x)
raw2=joinpath(r["rawdir"],"raw_planned.bin");write_raw(raw2,x,pp,pool)
h=open(sha256,raw2);g=U.Geometry(g.core,g.axis,g.length,g.area,h,g.net)
x=Pc.read_input(raw2);@assert U.validate_geometry(x,g,h)
m,_=Pc.prepare_model(x)
proof=Dict("record_sha256"=>r["record_sha256"],"method"=>cfg["method"],"slot"=>x.core,
    "sample"=>r["sample"],"contact_angle_deg"=>40,"raw_sha256"=>bytes2hex(h),
    "geometry_unplanned_sha256"=>bytes2hex(open(sha256,r["geometry"])),
    "entry_pressure_pa"=>minimum(m.entry[m.inlet]),"clay_endpoint"=>.8,"sand_endpoint"=>.95,
    "pressure_seed_dictionary"=>"zero only; new core endpoint-derived grid",
    "maximum_local_endpoint_pa"=>maximum(m.endpoint),"maximum_pressure_pa"=>pp[end],"passed"=>true)
@assert TOML.parsefile(cfg["prior_input"])==proof "Exact original input mismatch"
jsonfile(joinpath(cfg["out"],"INPUT_MATCH.json"),proof)

# Restore only the failed gas state; successful phase calls are never rerun.
paths=Pc.prepare_paths(m);state=Pc.initial(m.n);w=Pc.Workspace(m.n)
for j in 1:cfg["failed_sample"]
    Pc.advance_pc!(m,state,m.pressures[j],w,paths)
end
observation=Pc.observe(m,state,w)
p,A,rhs,keep,_=U.Flow.prepare(g.net,w.kg)
@assert p.n>0
signature=U.Flow.parts_hash(p,keep)
serialize(joinpath(cfg["out"],"failed_operator.jls"),(p,A,rhs,keep))
jsonfile(joinpath(cfg["out"],"CAPTURE.json"),Dict(
    "key"=>r["key"],"phase"=>"gas","sample"=>cfg["failed_sample"],
    "state_hash"=>digest_state(state,w),"sg"=>observation.sg_physical,
    "spanning_cells"=>p.n,"parts_hash"=>signature,"edges"=>length(p.tt),
    "minimum_conductance"=>minimum(p.tt),"maximum_conductance"=>maximum(p.tt),
    "operator_sha256"=>bytes2hex(open(sha256,joinpath(cfg["out"],"failed_operator.jls"))),
    "earlier_flow_calls_repeated"=>0,"production_writes"=>0))
println("CAPTURED ",r["key"]," gas sample=",cfg["failed_sample"]," spanning=",p.n);flush(stdout)
start=time()
d=StrictProbe.two_checks(U.Flow,p,A;deadline=time()+1800,reference_extra=true)
@assert U.Flow.parts_hash(p,keep)==signature
d["seconds"]=time()-start;d["key"]=r["key"];d["parts_hash"]=signature
d["scientific_acceptance"]=false;d["diagnostic_only"]=true
d["earlier_flow_calls_repeated"]=0;d["production_writes"]=0
jsonfile(joinpath(cfg["out"],"PROBE_RESULT.json"),d)
println("PROBE_FINISHED passed=",d["passed"]," seconds=",d["seconds"]);flush(stdout)
