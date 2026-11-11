module AdaptiveSelector
export select_adaptive

function select_adaptive(s, query; budget=10)
    @assert length(s) >= 2 && all(isfinite,s) && all(diff(s) .>= 0)
    @assert budget >= 2
    ids = vcat([1], [i for i in 2:length(s) if s[i] > s[i-1]])
    if length(ids) == 1
        ids = [1,length(s)]
    else
        ids[end] = length(s)
    end
    values = Dict{Int,Tuple{Float64,Float64}}()
    order = Int[]
    function observe(rank)
        @assert !haskey(values,rank) && length(order) < budget
        v = query(ids[rank])
        @assert length(v) == 2 && all(isfinite,v)
        values[rank] = (Float64(v[1]),Float64(v[2]))
        push!(order,ids[rank])
    end
    observe(1); observe(length(ids))
    while length(values) < min(budget,length(ids))
        known = sort(collect(keys(values)))
        intervals = Tuple{Float64,Float64,Int,Int,Int}[]
        for (a,b) in zip(known[1:end-1],known[2:end])
            b-a > 1 || continue
            delta = max(abs(values[b][1]-values[a][1]),abs(values[b][2]-values[a][2]))
            push!(intervals,(delta,s[ids[b]]-s[ids[a]],-a,a,b))
        end
        _,_,_,a,b = maximum(intervals)
        observe(fld(a+b,2))
    end
    [ids[i] for i in sort(collect(keys(values)))],order
end
end
