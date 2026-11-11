module SelectStates
export select_states

"""Ordered nearest-state matching to uniform saturation targets; no new MIP states."""
function select_states(s::AbstractVector{<:Real}, budget::Int=10)
    @assert budget >= 2 && length(s) >= 2
    @assert all(isfinite, s) && all(diff(s) .>= 0)
    # Equal saturations provide no extra interpolation knots. Retain both range ends.
    ids = [1; findall(diff(s) .> 0) .+ 1]
    if length(ids) == 1
        return [1, length(s)], [Float64(s[1]), Float64(s[end])]
    end
    ids[end] = length(s)
    count = min(budget, length(ids))
    targets = collect(range(Float64(s[1]), Float64(s[end]); length=count))
    n = length(ids)
    costs = fill(Inf, count, n)
    parent = zeros(Int, count, n)
    costs[1, 1] = 0.0
    scale = Float64(s[end] - s[1])
    for j in 2:count
        for i in j:(n-count+j)
            j == count && i != n && continue
            # Strict < chooses the lower predecessor on exact ties.
            for k in (j-1):(i-1)
                v = costs[j-1, k] + ((s[ids[i]]-targets[j])/scale)^2
                if v < costs[j, i]
                    costs[j, i] = v
                    parent[j, i] = k
                end
            end
        end
    end
    @assert isfinite(costs[count, n])
    positions = zeros(Int, count); positions[end] = n
    for j in count:-1:2
        positions[j-1] = parent[j, positions[j]]
    end
    chosen = ids[positions]
    @assert chosen[1] == 1 && chosen[end] == length(s)
    @assert all(diff(chosen) .> 0) && all(diff(s[chosen]) .> 0)
    chosen, targets
end
end
