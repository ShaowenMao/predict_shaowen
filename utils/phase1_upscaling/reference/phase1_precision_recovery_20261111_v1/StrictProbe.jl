module StrictProbe
using LinearAlgebra, SparseArrays

"""Unchanged strict acceptance; extra accuracy/work only in the correction solve."""
function reached(metrics, mass, tight)
    target = tight ? 1e-11 : 1e-9
    all(isfinite, metrics) && isfinite(mass) && metrics[3] > 0 &&
        metrics[7] >= -1e-9 && metrics[8] <= 1+1e-9 &&
        metrics[1] <= min(1e-10, target) &&
        max(metrics[5], metrics[6], mass) <= target
end

function repair(flow, p, A; shift, tight, deadline, inner_tol=1e-9, maxouter=80)
    B = Diagonal(p.scale)*A*Diagonal(p.scale)
    M = cholesky(Symmetric((B+B')/2 + shift*I))
    op = flow.QuadOperator(p)
    pressure = hcat(fill(0.5, p.n), zeros(p.n))
    history = Vector{Float64}[]
    for k in 0:maxouter
        r, metrics = flow.audit(p, pressure)
        mass = norm(r,1)/max(abs(metrics[3]), floatmin(Float64))
        push!(history, vcat(Float64(k), metrics, mass))
        if reached(metrics, mass, tight)
            @assert flow.gates(metrics)
            return (; passed=true, q=metrics[3], metrics, history, pressure)
        end
        (k == maxouter || time() > deadline) && break
        b = p.scale .* r
        bn = norm(b)
        @assert isfinite(bn) && bn > 0
        restart = min(50, p.n)
        dy = flow.gmres(op, b/bn; Pl=flow.Prec(M), restart=restart,
                       maxiter=8restart, reltol=inner_tol, abstol=0.0)
        @assert all(isfinite,dy)
        pressure = flow.pressure_add(pressure, p.scale .* (dy*bn))
    end
    (; passed=false, q=last(history)[4], metrics=last(history)[2:9], history, pressure)
end

function two_checks(flow, p, A; deadline, reference_extra=false)
    attempts = Any[]
    chosen = Float64[]
    solutions = Any[]
    for tight in (false, true)
        success = false
        shifts = tight && reference_extra ? (1e-14,1e-13,1e-12,1e-15) : (1e-14,1e-13,1e-12)
        for shift in shifts
            shift in chosen && continue
            time() > deadline && break
            start = time()
            try
                result = repair(flow, p, A; shift, tight, inner_tol=(shift==1e-15 ? 1e-11 : 1e-9),
                                deadline=min(deadline,time()+420))
                push!(attempts, Dict("shift"=>shift,"tight"=>tight,
                      "passed"=>result.passed,"seconds"=>time()-start,
                      "history"=>result.history))
                if result.passed
                    push!(chosen,shift);push!(solutions,result)
                    success = true
                    break
                end
            catch err
                err isa PosDefException || rethrow()
                push!(attempts, Dict("shift"=>shift,"tight"=>tight,
                      "passed"=>false,"seconds"=>time()-start,
                      "error"=>sprint(showerror,err)))
            end
        end
        if !success
            return Dict("passed"=>false,"attempts"=>attempts,"failed_stage"=>(tight ? "reference" : "primary"))
        end
    end
    q1, q2 = solutions[1].q, solutions[2].q
    difference = abs(q1-q2)/max(abs(q1),abs(q2),floatmin(Float64))
    Dict("passed"=>difference<=1e-5, "attempts"=>attempts,
         "primary_q"=>q1,"reference_q"=>q2,"difference"=>difference,
         "shifts"=>chosen,"metrics"=>solutions[2].metrics,
         "primary_mass"=>last(solutions[1].history)[10],
         "reference_mass"=>last(solutions[2].history)[10])
end
end
