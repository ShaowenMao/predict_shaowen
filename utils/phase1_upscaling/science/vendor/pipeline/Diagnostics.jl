module Diagnostics
const totals = Dict{String,Float64}()
const counts = Dict{String,Int}()
const receipt = Ref{Union{Nothing,String}}(nothing)
function measure(f, name)
    started = time_ns()
    try
        f()
    finally
        totals[name] = get(totals,name,0.0)+(time_ns()-started)/1e9
        counts[name] = get(counts,name,0)+1
    end
end
function source(f)
    if receipt[] === nothing
        measure(f,"source_verify")
    else
        counts["source_receipt_hits"] = get(counts,"source_receipt_hits",0)+1
        receipt[]
    end
end
function reset!(hash=nothing)
    empty!(totals); empty!(counts); receipt[]=hash
end
function verify_receipt!(f, expected)
    @assert f()==expected "Source changed during batch; reject the whole worker"
end
snapshot() = Dict("seconds"=>copy(totals),"counts"=>copy(counts),
    "source_verify_includes_nested_file_hash_time"=>true)
end
