"""Verify the pinned importer's epsilon shift without changing tables or evaluators."""
function endpoint_evidence(f, t)
    @assert all(isfinite, t)
    @assert t[1, 1] == 0.0 && t[1, 2] == 0.0
    x = copy(t[:, 1]); y = copy(t[:, 2])
    if x[end] < 1.0
        push!(x, 1.0); push!(y, y[end])
    end
    # Independent reproduction of the source-pinned ensure_endpoints! operation.
    for i in (length(x)-1):-1:2
        if y[i] != y[i-1]
            x[i] -= 1e-16
            break
        end
    end
    for i in 1:(length(x)-1)
        if y[i] != y[i+1]
            x[i] -= 1e-16
            break
        end
    end
    @assert f.k.X == x && f.k.F == y
    ix = Jutul.first_lower(x, 0.0)
    x0, x1 = x[ix:ix+1]; y0, y1 = y[ix:ix+1]
    @assert x0 == -1e-16 && y0 == 0.0 && x1 > 0.0 && y1 > 0.0
    slope = (y1-y0)/(x1-x0)
    prediction = y0 + slope*(0.0-x0)
    @assert 0.0 < prediction <= eps(Float64)
    @assert f(0.0) == prediction
    @assert ForwardDiff.derivative(f, 0.0) == slope
    Dict("native_krg_zero" => f(0.0), "predicted_native_krg_zero" => prediction,
         "active_interval_x" => [x0, x1], "active_interval_y" => [y0, y1],
         "active_slope" => slope, "raw_zero_preserved" => true,
         "legacy_strict_zero_passed" => false,
         "whole_imported_interpolator_matches_pinned_source" => true)
end
