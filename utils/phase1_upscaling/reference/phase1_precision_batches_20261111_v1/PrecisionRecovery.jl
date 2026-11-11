module PrecisionRecovery
using TOML, SHA, Serialization, LinearAlgebra
include("StrictProbe.jl")
include("AccurateAudit.jl")

function checks(flow,p,x,q,tight)
    @assert x isa Vector{BigFloat} && length(x)==p.n
    @assert all(v->isfinite(v) && precision(v)>=256,x)
    a=AccurateAudit.audit(p,x)
    b=AccurateAudit.independent(p,x)
    agreement=abs(b.metrics[3]-q)/max(abs(b.metrics[3]),abs(q),floatmin(Float64))
    old=flow.audit(p,AccurateAudit.pair(x))
    oldmass=norm(old[1],1)/max(abs(old[2][3]),floatmin(Float64))
    Dict("passed"=>StrictProbe.reached(a.metrics,a.mass,tight) &&
         StrictProbe.reached(b.metrics,b.mass,tight) && flow.gates(a.metrics) &&
         flow.gates(b.metrics) && agreement<=1e-5 && a.metrics[3]==q,
         "face_metrics"=>a.metrics,"face_mass"=>a.mass,
         "independent_metrics"=>b.metrics,"independent_mass"=>b.mass,
         "independent_q_difference"=>agreement,
         "original_binary128_passed"=>StrictProbe.reached(old[2],oldmass,tight),
         "original_binary128_mass"=>oldmass)
end

function reaudit(flow,p,saved)
    root=saved["precision"]
    @assert bytes2hex(open(sha256,joinpath(root,"RESULT.toml")))==saved["probe_sha256"]
    result=TOML.parsefile(joinpath(root,"RESULT.toml"))
    @assert result["passed"] && result["mode"]=="precision"
    @assert result["parts_hash"]==saved["parts_hash"]
    successful=filter(a->a["passed"],result["attempts"])
    @assert length(successful)==2 && [a["tight"] for a in successful]==[false,true]
    @assert successful[1]["shift"]!=successful[2]["shift"]
    audits=Any[]
    for (k,tight) in enumerate((false,true))
        shift=successful[k]["shift"]
        name=(tight ? "reference" : "primary")*"_"*string(shift)*"_pressure.jls"
        q=result[tight ? "reference_q" : "primary_q"]
        x=deserialize(joinpath(root,name))
        audit=checks(flow,p,x,q,tight)
        @assert audit["passed"] "Saved precise pressure failed unchanged audit"
        push!(audits,merge(audit,Dict("tight"=>tight,"pressure_sha256"=>bytes2hex(open(sha256,joinpath(root,name))))))
    end
    q1,q2=result["primary_q"],result["reference_q"]
    @assert q2==saved["probe_q"] && abs(q1-q2)/max(abs(q1),abs(q2))<=1e-5
    Dict("passed"=>true,"new_solves"=>0,"reference_q"=>q2,"audits"=>audits,
         "source_probe_sha256"=>saved["probe_sha256"],"accurate_bits"=>256,"independent_bits"=>384)
end

function two_checks(flow,p,A;deadline,out,prefix)
    attempts=Any[];chosen=Float64[];solutions=Any[];audits=Any[]
    for tight in (false,true)
        success=false
        for shift in (tight ? (1e-14,1e-13,1e-12,1e-15) : (1e-14,1e-13,1e-12))
            shift in chosen && continue
            time()>deadline && break
            start=time()
            try
                result=AccurateAudit.repair(flow,StrictProbe,p,A;shift,tight,
                    deadline=min(deadline,time()+420),inner_tol=(shift==1e-15 ? 1e-11 : 1e-9),maxouter=80)
                name=prefix*(tight ? "_reference_" : "_primary_")*string(shift)*"_pressure.jls"
                path=joinpath(out,name);@assert !ispath(path);serialize(path,result.pressure)
                push!(attempts,Dict("shift"=>shift,"tight"=>tight,"passed"=>result.passed,
                    "seconds"=>time()-start,"history"=>result.history,
                    "pressure_file"=>name,"pressure_sha256"=>bytes2hex(open(sha256,path))))
                if result.passed
                    audit=checks(flow,p,result.pressure,result.q,tight)
                    push!(audits,audit)
                    if !audit["passed"]
                        return Dict("passed"=>false,"failed_stage"=>"independent_audit","attempts"=>attempts,"audits"=>audits)
                    end
                    push!(chosen,shift);push!(solutions,result);success=true;break
                end
            catch err
                err isa PosDefException || rethrow()
                push!(attempts,Dict("shift"=>shift,"tight"=>tight,"passed"=>false,"error"=>sprint(showerror,err)))
            end
        end
        if !success
            return Dict("passed"=>false,"failed_stage"=>(tight ? "reference" : "primary"),"attempts"=>attempts,"audits"=>audits)
        end
    end
    q1,q2=solutions[1].q,solutions[2].q
    diff=abs(q1-q2)/max(abs(q1),abs(q2),floatmin(Float64))
    Dict("passed"=>diff<=1e-5,"difference"=>diff,"primary_q"=>q1,"reference_q"=>q2,
         "metrics"=>solutions[2].metrics,"attempts"=>attempts,"audits"=>audits,"shifts"=>chosen,
         "primary_target"=>1e-9,"reference_target"=>1e-11,"agreement_target"=>1e-5)
end
end
