module SavedPrimary
using TOML,SHA,Serialization
using ..PrecisionRecovery
const R=PrecisionRecovery

function exact_vector(x,n)
    if x isa AbstractMatrix{Float64}
        @assert size(x)==(n,2) && all(isfinite,x)
        return setprecision(BigFloat,384) do
            y=[BigFloat(x[k,1])+BigFloat(x[k,2]) for k in 1:n]
            for k in 1:n
                exact=setprecision(BigFloat,2048) do
                    BigFloat(x[k,1])+BigFloat(x[k,2])
                end
                @assert y[k]==exact "Pressure representation conversion must be exact"
            end
            y
        end
    end
    R.pressure_vector(x,n,false)
end

function durable(path,d)
    @assert !ispath(path)
    open(path,"w") do io
        TOML.print(io,d);flush(io);@assert ccall(:fsync,Cint,(Cint,),fd(io))==0
    end
end

function reference_only(flow,p;primary,expected_hash,original_q,out,solve_reference=R.DirectAudit.solve)
    @assert bytes2hex(open(sha256,primary))==expected_hash
    x=exact_vector(deserialize(primary),p.n)
    q1=R.AccurateAudit.audit(p,x;bits=512).metrics[3]
    a=R.checks(flow,p,x,q1,false)
    difference=abs(q1-original_q)/max(abs(q1),abs(original_q))
    @assert a["passed"] && difference<=1e-5 "Saved primary failed unchanged checks"
    name=joinpath(out,"primary_pressure.jls");@assert !ispath(name);serialize(name,x)
    first=Dict("passed"=>true,"reference"=>false,"algorithm"=>"saved_iterative",
        "bits"=>minimum(precision.(x)),"metrics"=>a["face_metrics"],"mass"=>a["face_mass"],
        "independent_metrics"=>a["independent_metrics"],"independent_mass"=>a["independent_mass"],
        "independent_difference"=>a["independent_q_difference"],"new_solves"=>0,
        "original_pressure_sha256"=>expected_hash,"pressure_sha256"=>bytes2hex(open(sha256,name)),
        "original_q"=>original_q,"original_q_difference"=>difference)
    durable(joinpath(out,"primary_AUDIT.toml"),first)
    # One independent reference only; no primary solver or shift-retry loop.
    d=solve_reference(p;bits=384,reference=true)
    name=joinpath(out,"reference_pressure.jls");@assert !ispath(name);serialize(name,d.pressure)
    b=R.checks(flow,p,d.pressure,d.metrics[3],true)
    second=Dict("passed"=>b["passed"],"reference"=>true,"algorithm"=>"direct_ldlt_reversed_amd",
        "bits"=>384,"metrics"=>b["face_metrics"],"mass"=>b["face_mass"],
        "independent_metrics"=>b["independent_metrics"],"independent_mass"=>b["independent_mass"],
        "independent_difference"=>b["independent_q_difference"],"new_solves"=>1,
        "pressure_sha256"=>bytes2hex(open(sha256,name)),"order_sha256"=>d.order_sha256,
        "seconds"=>d.seconds,"factor_nonzeros"=>d.factor_nonzeros,"minimum_pivot"=>d.minimum_pivot)
    durable(joinpath(out,"reference_AUDIT.toml"),second)
    q2=d.metrics[3];diff=abs(q1-q2)/max(abs(q1),abs(q2))
    @assert bytes2hex(open(sha256,primary))==expected_hash
    Dict("passed"=>b["passed"] && diff<=1e-5,"attempts"=>[first,second],"flows"=>[q1,q2],
        "difference"=>diff,"new_primary_solves"=>0,"new_reference_solves"=>1,
        "primary_target"=>1e-9,"reference_target"=>1e-11,"agreement_target"=>1e-5)
end

function reaudit(flow,p,saved)
    root=saved["precision"];file=joinpath(root,"RESULT.toml")
    @assert bytes2hex(open(sha256,file))==saved["probe_sha256"]
    d=TOML.parsefile(file)
    @assert d["passed"] && d["mode"]=="saved_primary_direct_reference" && d["parts_hash"]==saved["parts_hash"]
    @assert d["new_primary_solves"]==0 && d["new_reference_solves"]==1
    @assert [a["algorithm"] for a in d["attempts"]]==["saved_iterative","direct_ldlt_reversed_amd"]
    audits=Any[]
    for (j,tight) in enumerate((false,true))
        path=joinpath(root,tight ? "reference_pressure.jls" : "primary_pressure.jls")
        @assert bytes2hex(open(sha256,path))==d["attempts"][j]["pressure_sha256"]
        a=R.checks(flow,p,deserialize(path),d["flows"][j],tight)
        @assert a["passed"]
        push!(audits,a)
    end
    q1,q2=d["flows"]
    @assert q2==saved["probe_q"] && abs(q1-q2)/max(abs(q1),abs(q2))<=1e-5
    Dict("passed"=>true,"new_solves"=>0,"reference_q"=>q2,"audits"=>audits,
        "source_probe_sha256"=>saved["probe_sha256"],"face_audit_bits"=>512,"independent_bits"=>640)
end
end
