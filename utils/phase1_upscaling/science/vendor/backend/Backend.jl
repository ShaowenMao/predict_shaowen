module Backend
using SHA
const UMFPACK_SHA = "9043e3a9f65ffc5d4438d2e6c49d31bbc1c40426722b4c39311564f8ea052a82"
@assert string(VERSION)=="1.12.6"
@assert bytes2hex(open(sha256,joinpath(Sys.STDLIB,"SparseArrays/src/solvers/umfpack.jl")))==UMFPACK_SHA
include("ManagedLU.jl")
mutable struct Solver
    flow::Module
    brine::LUReuse.Cache
    gas::LUReuse.Cache
end
Solver(flow)=Solver(flow,LUReuse.Cache(),LUReuse.Cache())
function (s::Solver)(net,kr,label)
    @assert label in ("brine","gas")
    q,info=LUReuse.corrected(s.flow,net,kr,label=="brine" ? s.brine : s.gas)
    info["fallback_used"]=info["original_solver_rescue"] ? info["rescue_info"]["fallback_used"] : false
    q,info
end
function clear!(s)
    LUReuse.clear!(s.brine); LUReuse.clear!(s.gas)
end
reset!()=foreach(k->LUReuse.lifetime[k]=0,keys(LUReuse.lifetime))
function report()
    r=copy(LUReuse.lifetime)
    @assert r["numeric_created"]==r["numeric_released"]
    @assert r["symbolic_created"]==r["symbolic_released"]
    r
end
end
