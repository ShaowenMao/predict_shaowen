module LUReuse
using LinearAlgebra, SparseArrays

mutable struct Cache
    factor::Any
    keep::BitVector
    colptr::Vector{Int}
    rowval::Vector{Int}
    dimensions::Tuple{Int,Int}
end
Cache() = Cache(nothing, falses(0), Int[], Int[], (0,0))
const lifetime=Dict("numeric_created"=>0,"numeric_released"=>0,"symbolic_created"=>0,"symbolic_released"=>0)
function release_native!(f;symbolic)
    f===nothing && return
    # Factors are private to a sequential phase backend; no other consumer may hold them.
    lock(f.lock) do
        if f.numeric.p!=C_NULL
            Base.finalize(f.numeric)
            @assert f.numeric.p==C_NULL
            lifetime["numeric_released"]+=1
        end
        if symbolic && f.symbolic.p!=C_NULL
            Base.finalize(f.symbolic)
            @assert f.symbolic.p==C_NULL
            lifetime["symbolic_released"]+=1
        end
    end
end
function clear!(c)
    release_native!(c.factor;symbolic=true)
    c.factor=nothing;empty!(c.keep);empty!(c.colptr);empty!(c.rowval);c.dimensions=(0,0)
end
samepattern(c,A,keep) = c.factor!==nothing && c.dimensions==size(A) &&
    c.keep==keep && c.colptr==A.colptr && c.rowval==A.rowval

function info_time(f,name)
    lib=SparseArrays.LibSuiteSparse
    @assert isdefined(lib,name) "Installed UMFPACK timing constant missing"
    value=f.info[Int(getfield(lib,name))+1]
    @assert isfinite(value)&&value>=0
    value
end

function obtain!(c,A,keep)
    start=time_ns();hit=samepattern(c,A,keep)
    if hit
        # lu! reuses symbolic analysis, never the old numeric factorization.
        release_native!(c.factor; symbolic=false)
        lu!(c.factor,A;reuse_symbolic=true,check=true)
    else
        clear!(c)
        control=SparseArrays.UMFPACK.get_umfpack_control(Float64,Int)
        control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP]=2.0
        c.factor=lu(A;control=control,check=true)
        c.keep=copy(keep);c.colptr=copy(A.colptr);c.rowval=copy(A.rowval);c.dimensions=size(A)
    end
    lifetime["numeric_created"]+=1
    hit || (lifetime["symbolic_created"]+=1)
    @assert c.factor.control[SparseArrays.UMFPACK.JL_UMFPACK_IRSTEP]==2.0
    c.factor,Dict{String,Any}("symbolic_reused"=>hit,
        "factorization_seconds"=>(time_ns()-start)/1e9,
        "umfpack_symbolic_wall_seconds"=>(hit ? 0.0 : info_time(c.factor,:UMFPACK_SYMBOLIC_WALLTIME)),
        "umfpack_numeric_wall_seconds"=>info_time(c.factor,:UMFPACK_NUMERIC_WALLTIME))
end

function corrected(flow,net,kr,c)
    p,A,rhs,keep,times=flow.prepare(net,kr)
    info=Dict{String,Any}("spanning_cells"=>p.n,"original_solver_rescue"=>false,
                         "numeric_refactorized"=>p.n>0)
    if p.n==0
        clear!(c)
        return 0.0,merge(info,times,Dict("symbolic_reused"=>false,"metrics"=>zeros(8),"refinements"=>0))
    end
    try
        f,ft=obtain!(c,A,keep);merge!(info,ft)
        start=time_ns();x=f\rhs;@assert all(isfinite,x)
        info["linear_solve_seconds"]=(time_ns()-start)/1e9
        pressure=hcat(x,zeros(p.n));metrics=zeros(8);refinement=0
        audit_seconds=0.;correction_seconds=0.
        for iteration=0:8
            refinement=iteration;start=time_ns();r,metrics=flow.audit(p,pressure)
            audit_seconds+=(time_ns()-start)/1e9
            if all(isfinite,metrics)&&metrics[1]<=1e-10&&metrics[5]<=1e-9&&metrics[6]<=1e-9;break;end
            iteration==8&&break
            start=time_ns();dx=f\r;@assert all(isfinite,dx);pressure=flow.pressure_add(pressure,dx)
            correction_seconds+=(time_ns()-start)/1e9
        end
        flow.gates(metrics)||throw(flow.RepairRequired("Candidate conservative audit or pressure bounds failed"))
        merge!(info,Dict("metrics"=>metrics,"refinements"=>refinement,
            "audit_seconds"=>audit_seconds,"correction_seconds"=>correction_seconds))
        return metrics[3],merge(info,times)
    catch err
        err isa flow.RepairRequired || rethrow()
        clear!(c)
        start=time_ns();q,original=flow.conductance(net,kr)
        info["original_solver_rescue"]=true
        info["rescue_seconds"]=(time_ns()-start)/1e9
        info["rescue_info"]=original
        info["metrics"]=original["metrics"]
        q,merge(info,times)
    end
end
end
