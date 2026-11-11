module PhaseFlow
using LinearAlgebra,SparseArrays,SHA,Libdl
pushfirst!(LOAD_PATH,joinpath(@__DIR__,"third_party"))
using IterativeSolvers
export Net,Parts,prepare,conductance,audit,pressure_add,apply_operator,bridge!,parts_hash,readvec

const EVAL=Ref{Ptr{Cvoid}}();const ADD=Ref{Ptr{Cvoid}}();const ERROR=Ref{Ptr{Cvoid}}()
function bridge!(path)
    h=Libdl.dlopen(path);EVAL[]=Libdl.dlsym(h,:phase_eval);ADD[]=Libdl.dlsym(h,:phase_add);ERROR[]=Libdl.dlsym(h,:phase_error)
end
function check(code)
    code==0 || error(unsafe_string(ccall(ERROR[],Cstring,())))
end
struct Net
    n::Int;e::Matrix{Int};h1::Vector{Float64};h2::Vector{Float64}
    ci::Vector{Int};co::Vector{Int};hi::Vector{Float64};ho::Vector{Float64}
    row::Vector{Int};neighbors::Vector{Int};edgeids::Vector{Int}
end
function Net(n,e,h1,h2,ci,co,hi,ho)
    row=ones(Int,n+1)
    for u in e;@assert 1<=u<=n;row[u+1]+=1;end
    for k in 2:n+1;row[k]+=row[k-1]-1;end
    dest=zeros(Int,2size(e,1));eid=similar(dest);cursor=copy(row)
    for k in axes(e,1), (u,v) in ((e[k,1],e[k,2]),(e[k,2],e[k,1]))
        dest[cursor[u]]=v;eid[cursor[u]]=k;cursor[u]+=1
    end
    @assert all(isfinite,h1)&&all(h1.>=0)&&all(isfinite,h2)&&all(h2.>=0)
    @assert length(ci)==length(hi)&&length(co)==length(ho)
    Net(n,e,h1,h2,ci,co,hi,ho,row,dest,eid)
end
struct Parts
    n::Int;i::Vector{Float64};j::Vector{Float64};tt::Vector{Float64}
    ii::Vector{Float64};oo::Vector{Float64};tin::Vector{Float64};tout::Vector{Float64};scale::Vector{Float64}
end
readvec(io,T,n)=read!(io,Vector{T}(undef,n))
function audit(p::Parts,x)
    @assert size(x)==(p.n,2)
    r=zeros(p.n);m=zeros(8)
    code=ccall(EVAL[],Cint,(Cint,Int64,Int64,Int64,Int64,Ptr{Float64},Ptr{Float64},Ptr{Float64},
        Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64}),
        0,p.n,length(p.tt),length(p.tin),length(p.tout),p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout,p.scale,x,r,m)
    check(code);r,m
end
function pressure_add(x,dx)
    @assert size(x)==(length(dx),2)
    y=similar(x);check(ccall(ADD[],Cint,(Int64,Ptr{Float64},Ptr{Float64},Ptr{Float64}),length(dx),x,dx,y));y
end
function apply_operator(p::Parts,v)
    @assert length(v)==p.n
    y=zeros(p.n);unused=zeros(8)
    code=ccall(EVAL[],Cint,(Cint,Int64,Int64,Int64,Int64,Ptr{Float64},Ptr{Float64},Ptr{Float64},
        Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64},Ptr{Float64}),
        1,p.n,length(p.tt),length(p.tin),length(p.tout),p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout,p.scale,v,y,unused)
    check(code);y
end
function reachable(net,open,seeds)
    seen=falses(net.n);queue=Vector{Int}(undef,net.n);tail=0;head=1
    for u in seeds
        if !seen[u];tail+=1;queue[tail]=u;seen[u]=true;end
    end
    @inbounds while head<=tail
        u=queue[head];head+=1
        for k in net.row[u]:net.row[u+1]-1
            v=net.neighbors[k]
            if open[net.edgeids[k]]&&!seen[v];tail+=1;queue[tail]=v;seen[v]=true;end
        end
    end
    seen
end
function prepare(net,kr)
    started=time_ns();@assert length(kr)==net.n&&all(x->isfinite(x)&&0<=x<=1,kr)
    t=zeros(size(net.e,1));opened=falses(length(t))
    @inbounds for k in eachindex(t)
        a=net.h1[k]*kr[net.e[k,1]];b=net.h2[k]*kr[net.e[k,2]]
        opened[k]=a>0&&b>0
        if opened[k];t[k]=1.0/(1.0/a+1.0/b);end
    end
    ti=net.hi.*kr[net.ci];to=net.ho.*kr[net.co]
    inn=reachable(net,opened,net.ci[ti.>0]);out=reachable(net,opened,net.co[to.>0]);keep=inn.&out
    connectivity=(time_ns()-started)/1e9;started=time_ns()
    nn=count(keep);map=zeros(Int,net.n);map[keep]=1:nn
    use=opened.&keep[net.e[:,1]].&keep[net.e[:,2]]
    i=map[net.e[use,1]];j=map[net.e[use,2]];tt=t[use]
    ai=(ti.>0).&keep[net.ci];ao=(to.>0).&keep[net.co]
    ii=map[net.ci[ai]];oo=map[net.co[ao]];tin=ti[ai];tout=to[ao]
    A=sparse(vcat(i,j,i,j,ii,oo),vcat(i,j,j,i,ii,oo),vcat(tt,tt,-tt,-tt,tin,tout),nn,nn)
    rhs=zeros(nn);for k in eachindex(ii);rhs[ii[k]]+=tin[k];end
    d=diag(A);@assert all(x->isfinite(x)&&x>0,d)
    p=Parts(nn,Float64.(i),Float64.(j),tt,Float64.(ii),Float64.(oo),tin,tout,1.0./sqrt.(d))
    p,A,rhs,keep,Dict("connectivity_seconds"=>connectivity,"assembly_seconds"=>(time_ns()-started)/1e9)
end
function parts_hash(p,keep)
    ctx=SHA.SHA2_256_CTX()
    SHA.update!(ctx,reinterpret(UInt8,Int64[p.n,length(p.tt),length(p.tin),length(p.tout)]))
    SHA.update!(ctx,UInt8.(keep))
    for a in (p.i,p.j,p.tt,p.ii,p.oo,p.tin,p.tout);SHA.update!(ctx,reinterpret(UInt8,a));end
    bytes2hex(SHA.digest!(ctx))
end
struct RepairRequired <: Exception
    reason::String
end
struct CorrectionNotConverged <: Exception
    shift::Float64
    history::Vector
end
Base.showerror(io::IO,e::CorrectionNotConverged)=print(io,
    "Strict repair target not reached; no relaxed acceptance; preconditioner shift=",e.shift,
    "; final audit=",last(e.history))
function gates(m;repair=false)
    all(isfinite,m)&&m[1]<=1e-8&&m[5]<=1e-5&&m[6]<=1e-5&&m[3]>0&&m[7]>=-1e-7&&m[8]<=1+1e-7
end
function legacy(p,A,rhs,times)
    start=time_ns();control=SparseArrays.UMFPACK.get_umfpack_control(Float64,Int)
    control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP]=2.0
    factor=lu(A;control=control);times["factorization_seconds"]=(time_ns()-start)/1e9
    start=time_ns();x=factor\rhs;times["linear_solve_seconds"]=(time_ns()-start)/1e9
    @assert all(isfinite,x)
    pressure=hcat(x,zeros(p.n));metrics=zeros(8);refinement=0
    for iteration=0:8
        refinement=iteration;start=time_ns();r,metrics=audit(p,pressure)
        times["audit_seconds"]+=(time_ns()-start)/1e9
        if all(isfinite,metrics)&&metrics[1]<=1e-10&&metrics[5]<=1e-9&&metrics[6]<=1e-9;break;end
        iteration==8&&break
        start=time_ns();dx=factor\r;@assert all(isfinite,dx);pressure=pressure_add(pressure,dx)
        times["correction_seconds"]+=(time_ns()-start)/1e9
    end
    gates(metrics)||throw(RepairRequired("Conservative audit or pressure bounds failed"))
    metrics[3],metrics,refinement,pressure
end
struct QuadOperator <: AbstractMatrix{Float64}
    p::Parts
end
Base.size(a::QuadOperator)=(a.p.n,a.p.n)
LinearAlgebra.mul!(y::AbstractVector,a::QuadOperator,x::AbstractVector)=copyto!(y,apply_operator(a.p,x))
struct Prec{T}
    factor::T
end
LinearAlgebra.ldiv!(y::AbstractVector,p::Prec,x::AbstractVector)=copyto!(y,p.factor\x)
LinearAlgebra.ldiv!(p::Prec,x::AbstractVector)=copyto!(x,p.factor\x)
function repair(p,A;shift,tight)
    B=Diagonal(p.scale)*A*Diagonal(p.scale)
    M=cholesky(Symmetric((B+B')/2+shift*I));op=QuadOperator(p)
    pressure=hcat(fill(0.5,p.n),zeros(p.n));target=tight ? 1e-11 : 1e-9
    begin_time=time_ns();history=[];reached=false;metrics=zeros(8);outer=0
    for k in 0:24
        outer=k;r,metrics=audit(p,pressure);mass=norm(r,1)/max(abs(metrics[3]),floatmin(Float64))
        push!(history,vcat(Float64(k),metrics,mass))
        valid=all(isfinite,metrics)&&metrics[3]>0&&metrics[7]>=-1e-9&&metrics[8]<=1+1e-9
        reached=valid&&metrics[1]<=min(1e-10,target)&&max(metrics[5],metrics[6],mass)<=target
        reached&&break
        (k==24||(time_ns()-begin_time)/1e9>1800)&&break
        b=p.scale.*r;bn=norm(b);@assert isfinite(bn)&&bn>0
        restart=min(50,p.n)
        dy=gmres(op,b/bn;Pl=Prec(M),restart=restart,maxiter=8restart,reltol=1e-5,abstol=0.0)
        @assert all(isfinite,dy)
        pressure=pressure_add(pressure,p.scale.*(dy*bn))
    end
    reached||throw(CorrectionNotConverged(shift,history))
    @assert gates(metrics)
    metrics[3],metrics,outer,pressure,history
end
function checked_repair(p,A,shifts;tight)
    attempts=Any[]
    for shift in shifts
        try
            result=repair(p,A;shift=shift,tight=tight)
            push!(attempts,Dict("shift"=>shift,"passed"=>true))
            return result,shift,attempts
        catch e
            (e isa CorrectionNotConverged||e isa PosDefException)||rethrow()
            push!(attempts,Dict("shift"=>shift,"passed"=>false,"error"=>sprint(showerror,e)))
            println("CORRECTION_RETRY shift=",shift," ",sprint(showerror,e));flush(stdout)
        end
    end
    error("All strict correction preconditioners failed: ",attempts)
end
function conductance(net,kr;force_repair=false)
    p,A,rhs,keep,times=prepare(net,kr)
    for name in ("factorization_seconds","linear_solve_seconds","audit_seconds","correction_seconds","repair_seconds");times[name]=0.0;end
    info=Dict{String,Any}("spanning_cells"=>p.n,"fallback_used"=>false)
    if p.n==0;return 0.0,merge(info,times,Dict("metrics"=>zeros(8),"refinements"=>0));end
    try
        force_repair&&throw(RepairRequired("Explicit control exercise"))
        q,metrics,refinements,pressure=legacy(p,A,rhs,times)
        info["metrics"]=metrics;info["refinements"]=refinements
        return q,merge(info,times)
    catch e
        e isa RepairRequired||rethrow()
        start=time_ns()
        # Shifts affect only the approximate preconditioner, never the audited operator.
        primary,s1,a1=checked_repair(p,A,(1e-12,1e-14,1e-13);tight=false)
        q1,m1,_,_,h1=primary
        candidates=filter(x->x!=s1,(1e-14,1e-13,1e-12))
        reference,s2,a2=checked_repair(p,A,candidates;tight=true)
        q2,m2,refinements,_,h2=reference
        difference=abs(q1-q2)/max(abs(q1),abs(q2),floatmin(Float64))
        @assert difference<=1e-5 "Repair sensitivity unresolved"
        times["repair_seconds"]=(time_ns()-start)/1e9
        info["fallback_used"]=true;info["metrics"]=m2;info["refinements"]=refinements
        info["repair_difference"]=difference;info["repair_history"]=Dict("primary"=>h1,"reference"=>h2)
        info["repair_shifts"]=[s1,s2];info["repair_attempts"]=Dict("primary"=>a1,"reference"=>a2)
        return q2,merge(info,times)
    end
end
end
