module StandaloneMIP
using SHA
export Model, State, Workspace, read_model, read_state, initial, step!, advance!, phase!,
       same_state, state_digests, readvec, reach!, adjacency

const MAGIC = Int64(2026090901)
struct Curve
    sg::Vector{Float64}
    gas::Vector{Float64}
    brine::Vector{Float64}
    pc::Vector{Float64}
end
struct Model
    n::Int
    row::Vector{Int}
    neighbor::Vector{Int}
    inlet::Vector{Int}
    outlet::Vector{Int}
    material::Vector{UInt8}
    scale::Vector{Float64}
    entry::Vector{Float64}
    endpoint::Vector{Float64}
    pv::Vector{Float64}
    curves::Tuple{Curve,Curve}
    pressures::Vector{Float64}
    events::Vector{Float64}
end
mutable struct State
    pressure::Float64
    breakthrough::Float64
    sg::Vector{Float64}
    invaded::Vector{UInt8}
    trapped::Vector{UInt8}
end
struct Workspace
    target::Vector{Float64}
    oldsg::Vector{Float64}
    oldtrapped::Vector{UInt8}
    available::Vector{UInt8}
    mobile::Vector{UInt8}
    mask::Vector{UInt8}
    drain::Vector{UInt8}
    reached::Vector{UInt8}
    queue::Vector{Int}
    kw::Vector{Float64}
    kg::Vector{Float64}
end
Workspace(n::Int) = Workspace(zeros(n),zeros(n),zeros(UInt8,n),zeros(UInt8,n),
    zeros(UInt8,n),zeros(UInt8,n),zeros(UInt8,n),zeros(UInt8,n),zeros(Int,n),zeros(n),zeros(n))
initial(n::Int) = State(0.0,NaN,zeros(n),zeros(UInt8,n),zeros(UInt8,n))
readvec(io,T,n) = read!(io,Vector{T}(undef,n))

function adjacency(n,e)
    row=ones(Int,n+1)
    for u in e
        @assert 1<=u<=n
        row[u+1]+=1
    end
    for i in 2:n+1; row[i]+=row[i-1]-1; end
    neighbor=Vector{Int}(undef,row[end]-1); cursor=copy(row)
    for j in axes(e,1)
        u,v=e[j,1],e[j,2]
        neighbor[cursor[u]]=v;cursor[u]+=1
        neighbor[cursor[v]]=u;cursor[v]+=1
    end
    row,neighbor
end

function read_model(path)
    open(path,"r") do io
        magic,n,ne,ni,no,n1,n2,np=readvec(io,Int64,8)
        @assert magic==MAGIC && n>0 && ne>=0 && np>0
        e=reshape(readvec(io,Int64,2ne),ne,2)
        inlet=readvec(io,Int64,ni);outlet=readvec(io,Int64,no)
        material=readvec(io,UInt8,n)
        scale=readvec(io,Float64,n);entry=readvec(io,Float64,n)
        endpoint=readvec(io,Float64,n);pv=readvec(io,Float64,n)
        tables=map((n1,n2)) do nr
            t=reshape(readvec(io,Float64,4nr),nr,4)
            @assert nr>=2 && all(isfinite,t) && all(diff(t[:,1]).>0) && all(diff(t[:,4]).>0)
            @assert t[1,1]==0 && t[end,3]==0 && all(0 .<= t[:,2:3] .<= 1)
            Curve(t[:,1],t[:,2],t[:,3],t[:,4])
        end
        pressures=readvec(io,Float64,np)
        @assert eof(io) && all(isfinite,pressures) && all(diff(pressures).>=0)
        @assert all(x->x in (1,2),material) && all(x->isfinite(x)&&x>0,scale)
        @assert all(x->isfinite(x)&&x>0,entry) && all(endpoint.>=entry)
        @assert all(1 .<= inlet .<= n) && all(1 .<= outlet .<= n)
        row,neighbor=adjacency(n,e)
        Model(n,row,neighbor,inlet,outlet,material,scale,entry,endpoint,pv,tables,
              pressures,sort!(unique(vcat(entry,endpoint))))
    end
end

function read_state(io,n)
    pressure=read(io,Float64); breakthrough=read(io,Float64)
    State(pressure,breakthrough,readvec(io,Float64,n),readvec(io,UInt8,n),readvec(io,UInt8,n))
end

@inline function interval(x,q)
    @assert x[1]<=q<=x[end]
    k=min(searchsortedlast(x,q),length(x)-1)
    @inbounds w=(q-x[k])/(x[k+1]-x[k])
    k,w
end
@inline function weighted(y,k,w)
    # Match MATLAB's weighted-endpoint arithmetic; do not use muladd/fastmath.
    @inbounds (1.0-w)*y[k]+w*y[k+1]
end
@inline function interpolate(x,y,q)
    k,w=interval(x,q)
    weighted(y,k,w)
end

function phase!(m,s,w)
    @inbounds for i in 1:m.n
        c=m.curves[m.material[i]]
        k,weight=interval(c.sg,s.sg[i])
        w.kw[i]=weighted(c.brine,k,weight)
        w.kg[i]=weighted(c.gas,k,weight)
        @assert isfinite(w.kw[i]) && 0<=w.kw[i]<=1
        @assert isfinite(w.kg[i]) && 0<=w.kg[i]<=1
    end
end

function reach!(connected,row,neighbor,mask,seeds,queue)
    fill!(connected,0); head=1;tail=0
    @inbounds for s in seeds
        if mask[s]!=0 && connected[s]==0
            tail+=1;queue[tail]=s;connected[s]=1
        end
    end
    @inbounds while head<=tail
        u=queue[head];head+=1
        for j in row[u]:row[u+1]-1
            v=neighbor[j]
            if mask[v]!=0 && connected[v]==0
                connected[v]=1;tail+=1;queue[tail]=v
            end
        end
    end
    connected
end

function step!(m,s,pc,w)
    @assert isfinite(pc) && pc>=0 && pc>=s.pressure
    copyto!(w.oldsg,s.sg);copyto!(w.oldtrapped,s.trapped)
    @inbounds for i in 1:m.n
        c=m.curves[m.material[i]]
        w.available[i]=m.entry[i]<=pc
        p=clamp(pc/m.scale[i],c.pc[1],c.pc[end])
        v=interpolate(c.pc,c.sg,p)
        if pc>=m.endpoint[i]
            v=c.sg[end]
        elseif v>=c.sg[end]
            v=c.sg[end]-eps(c.sg[end])
        end
        @assert isfinite(v) && v>=0
        w.target[i]=v
    end
    phase!(m,s,w)
    @inbounds for i in 1:m.n
        w.mobile[i]=w.kw[i]>0;w.mask[i]=w.mobile[i]!=0 && s.trapped[i]==0
    end
    reach!(w.drain,m.row,m.neighbor,w.mask,m.outlet,w.queue)
    @inbounds for i in 1:m.n
        s.trapped[i]|=(w.mobile[i]!=0 && w.drain[i]==0)
        w.mask[i]=s.invaded[i]!=0 || (w.available[i]!=0 && w.drain[i]!=0)
    end
    reach!(w.reached,m.row,m.neighbor,w.mask,m.inlet,w.queue)
    @inbounds for i in 1:m.n
        if w.reached[i]!=0 && w.drain[i]!=0 && w.available[i]!=0
            s.sg[i]=w.target[i]
        end
        s.invaded[i]|=(w.reached[i]!=0 && w.available[i]!=0)
    end
    phase!(m,s,w)
    @inbounds for i in 1:m.n
        w.mobile[i]=w.kw[i]>0;w.mask[i]=w.mobile[i]!=0 && s.trapped[i]==0
    end
    reach!(w.drain,m.row,m.neighbor,w.mask,m.outlet,w.queue)
    @inbounds for i in 1:m.n
        s.trapped[i]|=(w.mobile[i]!=0 && w.drain[i]==0)
    end
    if isnan(s.breakthrough) && any(i->s.invaded[i]!=0,m.outlet);s.breakthrough=pc;end
    s.pressure=pc
    @inbounds for i in 1:m.n
        @assert s.sg[i]>=w.oldsg[i]-1e-12
        @assert w.oldtrapped[i]==0 || (s.sg[i]==w.oldsg[i] && s.trapped[i]!=0)
    end
    s
end

function advance!(m,s,pc,w)
    @assert pc>=s.pressure
    first=searchsortedlast(m.events,s.pressure)+1;last=searchsortedlast(m.events,pc)
    for i in first:last;step!(m,s,m.events[i],w);end
    if s.pressure<pc || pc==0;step!(m,s,pc,w);end
    max(0,last-first+1)
end

bitsame(a,b)=reinterpret(UInt64,a)==reinterpret(UInt64,b)
same_state(a,b)=a.pressure==b.pressure && isequal(a.breakthrough,b.breakthrough) &&
    bitsame(a.sg,b.sg) && a.invaded==b.invaded && a.trapped==b.trapped
state_digests(s)=(sha256(reinterpret(UInt8,s.sg)),sha256(vcat(s.invaded,s.trapped)))
end
