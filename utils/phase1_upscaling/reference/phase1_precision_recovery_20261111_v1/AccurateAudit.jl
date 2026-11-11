module AccurateAudit
using LinearAlgebra, SparseArrays

"""Same conservative face equations, with 256-bit pressure/residual arithmetic."""
function audit(p, pressure; bits=256)
    setprecision(BigFloat, bits) do
        x = BigFloat.(pressure)
        r = zeros(BigFloat, p.n)
        rhs = zeros(BigFloat, p.n)
        energy = BigFloat(0); qin = BigFloat(0); qout = BigFloat(0)
        for k in eachindex(p.tt)
            i, j = Int(p.i[k]), Int(p.j[k])
            delta = x[i] - x[j]
            flux = BigFloat(p.tt[k]) * delta
            r[i] -= flux; r[j] += flux
            energy += flux * delta
        end
        for k in eachindex(p.tin)
            i = Int(p.ii[k]); t = BigFloat(p.tin[k]); delta = 1 - x[i]
            flux = t * delta
            r[i] += flux; rhs[i] += t; qin += flux; energy += flux * delta
        end
        for k in eachindex(p.tout)
            i = Int(p.oo[k]); flux = BigFloat(p.tout[k]) * x[i]
            r[i] -= flux; qout += flux; energy += flux * x[i]
        end
        metrics = Float64[norm(r)/norm(rhs), qin, qout, energy,
            abs(qin-qout)/max(abs(qin),abs(qout)), abs(energy-qout)/abs(qout),
            minimum(x), maximum(x)]
        mass = Float64(sum(abs, r)/max(abs(qout), BigFloat(floatmin(Float64))))
        (; residual=Float64.(r), metrics, mass)
    end
end

"""Independent 384-bit assembled-operator audit, including boundary energy."""
function independent(p, pressure; bits=384)
    setprecision(BigFloat, bits) do
        x = BigFloat.(pressure)
        i, j, ii, oo = Int.(p.i), Int.(p.j), Int.(p.ii), Int.(p.oo)
        t, ti, to = BigFloat.(p.tt), BigFloat.(p.tin), BigFloat.(p.tout)
        A = sparse(vcat(i,j,i,j,ii,oo), vcat(i,j,j,i,ii,oo),
                   vcat(t,t,-t,-t,ti,to), p.n, p.n)
        rhs = zeros(BigFloat, p.n)
        for k in eachindex(ii); rhs[ii[k]] += ti[k]; end
        residual = rhs - A*x
        qin = sum(ti[k]*(1-x[ii[k]]) for k in eachindex(ti))
        qout = sum(to[k]*x[oo[k]] for k in eachindex(to))
        energy = dot(x, A*x) - 2dot(rhs, x) + sum(ti)
        metrics = Float64[norm(residual)/norm(rhs), qin, qout, energy,
            abs(qin-qout)/max(abs(qin),abs(qout)), abs(energy-qout)/abs(qout),
            minimum(x), maximum(x)]
        mass = Float64(sum(abs, residual)/max(abs(qout), BigFloat(floatmin(Float64))))
        (; residual=Float64.(residual), metrics, mass)
    end
end

function add(pressure, dx)
    setprecision(BigFloat, 256) do
        [pressure[k] + BigFloat(dx[k]) for k in eachindex(dx)]
    end
end

function pair(pressure)
    setprecision(BigFloat, 384) do
        high = Float64.(pressure)
        hcat(high, [Float64(BigFloat(pressure[k])-BigFloat(high[k])) for k in eachindex(high)])
    end
end

function repair(flow, gate, p, A; shift, tight, deadline, inner_tol=1e-9, maxouter=80)
    B = Diagonal(p.scale)*A*Diagonal(p.scale)
    M = cholesky(Symmetric((B+B')/2 + shift*I))
    op = flow.QuadOperator(p)
    pressure = setprecision(BigFloat, 256) do; fill(BigFloat(0.5), p.n); end
    history = Vector{Float64}[]
    for k in 0:maxouter
        a = audit(p, pressure)
        push!(history, vcat(Float64(k), a.metrics, a.mass))
        if gate.reached(a.metrics, a.mass, tight)
            @assert flow.gates(a.metrics)
            return (; passed=true, q=a.metrics[3], metrics=a.metrics, history, pressure)
        end
        (k == maxouter || time() > deadline) && break
        b = p.scale .* a.residual
        bn = norm(b)
        @assert isfinite(bn) && bn > 0
        restart = min(50, p.n)
        dy = flow.gmres(op, b/bn; Pl=flow.Prec(M), restart=restart,
                       maxiter=8restart, reltol=inner_tol, abstol=0.0)
        @assert all(isfinite, dy)
        pressure = add(pressure, p.scale .* (dy*bn))
    end
    (; passed=false, q=last(history)[4], metrics=last(history)[2:9], history, pressure)
end
end
