module ClayPc40
using DelimitedFiles
export reference, host_pc, interpolate_bounded, clay_table, settings40

function reference(root, sample, use)
    @assert sample in (4, 6) && use in ("host", "master")
    a = readdlm(joinpath(root, "data", "$(use)_sample$(sample).csv"), ','; skipstart=1)
    @assert a[1,1] == 0 && a[end,1] == (use == "host" ? .57 : .8)
    @assert all(isfinite, a) && all(diff(a[:,1]) .> 0)
    @assert a[1,2] > 0 && all(diff(a[:,2]) .> 0)
    a
end

function interpolate_bounded(x, y, s)
    x[1] <= s <= x[end] || throw(DomainError(s, "Outside tabulated saturation range"))
    i = min(searchsortedlast(x,s), length(x)-1)
    y[i] + (s-x[i])/(x[i+1]-x[i])*(y[i+1]-y[i])
end

# Host endpoint extension matches the existing constant-endpoint convention.
host_pc(t, sg) = interpolate_bounded(t[:,1],t[:,2],clamp(sg,t[1,1],t[end,1]))

function clay_table(old, master)
    @assert old[1,1] == 0 && old[end,1] == .8
    # Retain every old Kr knot and every new Pc knot, preserving both interpolants.
    sg = sort!(unique(vcat(old[:,1], master[:,1])))
    hcat(sg,
         [interpolate_bounded(old[:,1],old[:,2],s) for s in sg],
         [interpolate_bounded(old[:,1],old[:,3],s) for s in sg],
         [interpolate_bounded(master[:,1],master[:,2],s) for s in sg])
end

function settings40(P, old)
    @assert old.clay_endpoint == .8 && old.sand_endpoint == .95
    P.Settings((f == :co2_brine_theta_deg ? 40. : getfield(old,f)
                for f in fieldnames(P.Settings))...)
end
end
