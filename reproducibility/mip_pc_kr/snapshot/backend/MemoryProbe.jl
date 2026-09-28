module MemoryProbe
using Libdl
function snapshot()
    values=Dict{String,Any}()
    for line in eachline("/proc/self/status")
        key=first(split(line,':'))
        key in ("VmRSS","VmHWM","RssAnon","RssFile","RssShmem") || continue
        values[key*"_kib"]=parse(Int,split(line)[2])
    end
    for line in eachline("/proc/self/smaps_rollup")
        key=first(split(line,':'))
        key in ("Private_Clean","Private_Dirty","Anonymous","Swap") || continue
        values[key*"_kib"]=parse(Int,split(line)[2])
    end
    @assert isdefined(Base,:gc_live_bytes)
    values["julia_live_bytes"]=Base.gc_live_bytes()
    values
end
function after_gc()
    start=time_ns();GC.gc(true);result=snapshot()
    result["gc_seconds"]=(time_ns()-start)/1e9
    result
end
function terminal_trim()
    before=after_gc();handle=Libdl.dlopen("libc.so.6")
    symbol=Libdl.dlsym_e(handle,:malloc_trim)
    status=symbol==C_NULL ? -1 : ccall(symbol,Cint,(Csize_t,),0)
    Dict("before"=>before,"available"=>symbol!=C_NULL,"returncode"=>status,
         "after"=>snapshot(),"terminal_diagnostic_only"=>true)
end
end
