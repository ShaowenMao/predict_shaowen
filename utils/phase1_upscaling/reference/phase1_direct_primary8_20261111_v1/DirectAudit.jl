module DirectAudit
using LinearAlgebra, SparseArrays, SHA
pushfirst!(LOAD_PATH,joinpath(@__DIR__,"vendor"))
using AMD, LDLFactorizations
include("AccurateAudit.jl")

"""Assemble the unchanged face equations before rounding any diagonal sums."""
function assemble(p)
    i,j,ii,oo=Int.(p.i),Int.(p.j),Int.(p.ii),Int.(p.oo)
    t,ti,to=BigFloat.(p.tt),BigFloat.(p.tin),BigFloat.(p.tout)
    @assert all(isfinite,t) && all(t.>0) && all(ti.>0) && all(to.>0)
    A=sparse(vcat(i,j,i,j,ii,oo),vcat(i,j,j,i,ii,oo),vcat(t,t,-t,-t,ti,to),p.n,p.n)
    b=zeros(BigFloat,p.n)
    for k in eachindex(ii);b[ii[k]]+=ti[k];end
    @assert issymmetric(A) && all(diag(A).>0)
    A,b
end

function solve(p;bits=256,reference=false,maxfill=20_000_000)
    @assert bits>=256
    setprecision(BigFloat,bits) do
        t=time();A,b=assemble(p)
        scale=inv.(sqrt.(diag(A)))
        B=copy(A)
        for j in 1:p.n,k in nzrange(B,j)
            B.nzval[k]*=scale[B.rowval[k]]*scale[j]
        end
        @assert issymmetric(B)
        # Reverse relabeling gives the reference a separately ordered factorization.
        graph=SparseMatrixCSC(p.n,p.n,copy(A.colptr),copy(A.rowval),ones(nnz(A)))
        labels=reference ? collect(p.n:-1:1) : collect(1:p.n)
        order=labels[AMD.amd(graph[labels,labels])]
        F=LDLFactorizations.ldl_analyze(B;P=order)
        @assert length(F.Lx)<=maxfill "Factor fill exceeds isolated memory guard"
        @assert F.r1==0 && F.r2==0 && F.tol==0 "Regularization forbidden"
        LDLFactorizations.ldl_factorize!(B,F)
        @assert LDLFactorizations.factorized(F) && all(F.d.>0)
        pressure=scale.*(F\(scale.*b))
        @assert all(isfinite,pressure)
        a=AccurateAudit.audit(p,pressure;bits=max(bits,384))
        independent=AccurateAudit.independent(p,pressure;bits=max(bits+128,512))
        (;pressure,metrics=a.metrics,mass=a.mass,independent,
          seconds=time()-t,bits,reference,factor_nonzeros=length(F.Lx),
          order_sha256=bytes2hex(sha256(reinterpret(UInt8,order))),
          minimum_pivot=Float64(minimum(F.d)),new_solves=1)
    end
end
end
