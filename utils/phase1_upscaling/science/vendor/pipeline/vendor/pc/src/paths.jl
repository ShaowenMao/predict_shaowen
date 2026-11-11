# This computes the same minimax/maximin threshold values as the frozen spanning-tree
# construction. Heap keys are input threshold values, not approximated pressures.
function path_thresholds(m,threshold,seeds;gas)
    values=fill(gas ? Inf : -Inf,m.n)
    heap=Tuple{Float64,Int}[]
    function put!(key,node)
        push!(heap,(key,node));j=length(heap)
        while j>1
            p=j>>1;heap[p]<=heap[j]&&break
            heap[p],heap[j]=heap[j],heap[p];j=p
        end
    end
    function take!()
        first=heap[1];last=pop!(heap)
        if !isempty(heap)
            heap[1]=last;j=1
            while 2j<=length(heap)
                k=2j
                if k<length(heap)&&heap[k+1]<heap[k];k+=1;end
                heap[j]<=heap[k]&&break
                heap[j],heap[k]=heap[k],heap[j];j=k
            end
        end
        first
    end
    for u in unique(seeds)
        values[u]=threshold[u];put!(gas ? values[u] : -values[u],u)
    end
    while !isempty(heap)
        key,u=take!();value=gas ? key : -key
        value==values[u]||continue
        for j in m.row[u]:m.row[u+1]-1
            v=m.neighbor[j];candidate=gas ? max(value,threshold[v]) : min(value,threshold[v])
            if gas ? candidate<values[v] : candidate>values[v]
                values[v]=candidate;put!(gas ? candidate : -candidate,v)
            end
        end
    end
    values
end

function prepare_paths(m;backend=:auto)
    @assert backend in (:auto,:literal)
    backend==:literal&&return nothing
    maximum(m.entry)<minimum(m.endpoint)||return nothing
    @assert all(i->all(m.curves[i].brine[1:end-1].>0)&&m.curves[i].brine[end]==0,unique(m.material))
    connected=zeros(UInt8,m.n)
    reach!(connected,m.row,m.neighbor,ones(UInt8,m.n),[1],zeros(Int,m.n))
    all(!=(0),connected)||return nothing
    arrival=path_thresholds(m,m.entry,m.inlet;gas=true)
    loss=path_thresholds(m,m.endpoint,m.outlet;gas=false)
    @assert all(isfinite,arrival)&&all(isfinite,loss)
    @assert all(arrival.>=m.entry)&&all(loss.<=m.endpoint)&&maximum(arrival)<minimum(loss)
    (arrival=arrival,loss=loss,breakthrough=minimum(arrival[m.outlet]))
end

function advance_pc!(m,s,pc,w,paths)
    paths===nothing&&return advance!(m,s,pc,w)
    @assert isfinite(pc)&&pc>=s.pressure&&pc>=0
    count=max(0,searchsortedlast(m.events,pc)-searchsortedlast(m.events,s.pressure))
    for i in 1:m.n
        c=m.curves[m.material[i]];physical=min(pc,paths.loss[i])
        p=clamp(physical/m.scale[i],c.pc[1],c.pc[end])
        value=StandaloneMIP.interpolate(c.pc,c.sg,p)
        if physical>=m.endpoint[i]
            value=c.sg[end]
        elseif value>=c.sg[end]
            value=c.sg[end]-eps(c.sg[end])
        end
        invaded=paths.arrival[i]<=pc
        trapped=paths.loss[i]<=pc && paths.loss[i]<m.endpoint[i]
        value=invaded ? value : 0.0
        @assert value>=s.sg[i]-1e-12 && (!trapped||invaded)
        @assert s.trapped[i]==0 || (trapped&&value==s.sg[i])
        s.sg[i]=value;s.invaded[i]=invaded;s.trapped[i]=trapped
    end
    s.pressure=pc
    s.breakthrough=pc>=paths.breakthrough ? paths.breakthrough : NaN
    count
end
