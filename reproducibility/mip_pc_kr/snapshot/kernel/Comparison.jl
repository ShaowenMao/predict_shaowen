# Only scalar pore-volume reductions may differ by bounded IEEE roundoff.
# Every microscopic saturation/mobility/trapping hash remains bitwise exact.
same_average(x, y) = isfinite(x) && isfinite(y) && x >= 0 && y >= 0 &&
    abs(x-y) <= 32*eps(Float64)*max(abs(x),abs(y))

function compare_pc_lines(actual, reference)
    @assert length(actual)==length(reference)==101
    @assert actual[1]==reference[1]
    names=split(actual[1],',')
    aggregates=("sg_physical","trapped_pv_fraction","immobile_pv_fraction","invaded_pv_fraction")
    maxdiff=0.0
    for (a,b) in zip(actual[2:end],reference[2:end])
        x=split(a,','); y=split(b,',')
        @assert length(x)==length(y)==length(names)
        for (k,name) in enumerate(names)
            if name in aggregates
                u=parse(Float64,x[k]);v=parse(Float64,y[k])
                @assert same_average(u,v) "Pore-volume reduction differs beyond roundoff: $name"
                maxdiff=max(maxdiff,abs(u-v))
            else
                @assert x[k]==y[k] "MIP pressure, event, breakthrough or microscopic state changed: $name"
            end
        end
    end
    maxdiff
end
