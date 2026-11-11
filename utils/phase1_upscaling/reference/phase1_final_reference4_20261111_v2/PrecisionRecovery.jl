module PrecisionRecovery
using TOML,SHA,Serialization,LinearAlgebra
include("StrictProbe.jl");include("DirectAudit.jl")
const AccurateAudit=DirectAudit.AccurateAudit

function pressure_vector(x,n,tight)
    @assert x isa AbstractVector{BigFloat} && length(x)==n
    @assert all(v->isfinite(v) && precision(v)>=(tight ? 384 : 256),x)
    collect(x)
end

function checks(flow,p,x,q,tight)
    x=pressure_vector(x,p.n,tight)
    a=AccurateAudit.audit(p,x;bits=512)
    b=AccurateAudit.independent(p,x;bits=640)
    agreement=abs(b.metrics[3]-q)/max(abs(b.metrics[3]),abs(q),floatmin(Float64))
    old=flow.audit(p,AccurateAudit.pair(x))
    oldmass=norm(old[1],1)/max(abs(old[2][3]),floatmin(Float64))
    Dict("passed"=>StrictProbe.reached(a.metrics,a.mass,tight) &&
        StrictProbe.reached(b.metrics,b.mass,tight) && flow.gates(a.metrics) && flow.gates(b.metrics) &&
        agreement<=1e-5 && a.metrics[3]==q,"face_metrics"=>a.metrics,"face_mass"=>a.mass,
        "independent_metrics"=>b.metrics,"independent_mass"=>b.mass,"independent_q_difference"=>agreement,
        "original_binary128_passed"=>StrictProbe.reached(old[2],oldmass,tight),"original_binary128_mass"=>oldmass)
end

function reaudit(flow,p,saved)
    root=saved["precision"];file=joinpath(root,"RESULT.toml")
    @assert bytes2hex(open(sha256,file))==saved["probe_sha256"]
    d=TOML.parsefile(file)
    @assert d["passed"] && d["mode"]=="direct" && d["parts_hash"]==saved["parts_hash"]
    @assert length(d["attempts"])==2 && [a["reference"] for a in d["attempts"]]==[false,true]
    @assert d["attempts"][1]["order_sha256"]!=d["attempts"][2]["order_sha256"]
    audits=Any[]
    for (j,tight) in enumerate((false,true))
        name=(tight ? "reference" : "primary")*"_pressure.jls"
        x=deserialize(joinpath(root,name));a=checks(flow,p,x,d["flows"][j],tight)
        @assert a["passed"] "Saved direct pressure failed unchanged scientific checks"
        push!(audits,merge(a,Dict("tight"=>tight,"pressure_sha256"=>bytes2hex(open(sha256,joinpath(root,name))))))
    end
    q1,q2=d["flows"]
    @assert q2==saved["probe_q"] && abs(q1-q2)/max(abs(q1),abs(q2))<=1e-5
    Dict("passed"=>true,"new_solves"=>0,"reference_q"=>q2,"audits"=>audits,
        "source_probe_sha256"=>saved["probe_sha256"],"face_audit_bits"=>512,"independent_bits"=>640)
end

function two_checks(flow,p,A;deadline,out,prefix)
    attempts=Any[];audits=Any[];qs=Float64[]
    for tight in (false,true)
        @assert time()<deadline "New call time budget exhausted"
        d=DirectAudit.solve(p;bits=tight ? 384 : 256,reference=tight)
        name=prefix*(tight ? "_reference" : "_primary")*"_pressure.jls"
        path=joinpath(out,name);@assert !ispath(path);serialize(path,d.pressure)
        a=checks(flow,p,d.pressure,d.metrics[3],tight);push!(audits,a)
        push!(attempts,Dict("passed"=>a["passed"],"reference"=>tight,"bits"=>d.bits,"seconds"=>d.seconds,
            "metrics"=>d.metrics,"mass"=>d.mass,"independent_metrics"=>d.independent.metrics,
            "independent_mass"=>d.independent.mass,"order_sha256"=>d.order_sha256,
            "pressure_file"=>name,"pressure_sha256"=>bytes2hex(open(sha256,path))))
        a["passed"] || return Dict("passed"=>false,"attempts"=>attempts,"audits"=>audits)
        push!(qs,d.metrics[3])
    end
    q1,q2=qs;diff=abs(q1-q2)/max(abs(q1),abs(q2))
    Dict("passed"=>diff<=1e-5,"difference"=>diff,"primary_q"=>q1,"reference_q"=>q2,"flows"=>qs,
        "metrics"=>attempts[2]["metrics"],"attempts"=>attempts,"audits"=>audits,
        "primary_target"=>1e-9,"reference_target"=>1e-11,"agreement_target"=>1e-5)
end
end
