using DelimitedFiles, SHA, TOML
include(ARGS[1])
const P=FaultCorePc

function main()
    jobs,_=readdlm(ARGS[2],',',String;header=true)
    @assert size(jobs)==(522,5)
    @assert !ispath(ARGS[3])
    entries=open(ARGS[3],"w")
    println(entries,"slot,entry_pressure_pa,pc_recomputed,control_max_abs_sg_error")
    try
        for j=1:522
            slot=parse(Int,jobs[j,1]);@assert slot==j
            raw=jobs[j,2];curve=jobs[j,3];accepted=jobs[j,4]=="1";output=jobs[j,5]
            before=bytes2hex(open(sha256,raw))
            input=P.read_input(raw);@assert input.core==slot && length(input.pressures)==100
            model,_=P.prepare_model(input)
            entry=minimum(model.entry[model.inlet])
            data=accepted ? readdlm(curve,',',Float64;skipstart=1)[:,1:2] : zeros(100,2)
            @assert !accepted || data[:,1]==input.pressures
            # One independent Pc-only control per scenario verifies equivalence
            # before recovering any missing Pc without unnecessary Kr solves.
            err=0.0
            if !accepted || j==1
                result=P.upscale(model)
                recovered=hcat([r.pressure_pa for r in result.rows],[r.sg_physical for r in result.rows])
                if accepted
                    @assert recovered[:,1]==data[:,1]
                    err=maximum(abs.(recovered[:,2].-data[:,2]))
                    @assert err<=1e-13 "Pc-only engine does not match accepted coupled MIP"
                else
                    data=recovered
                end
            end
            @assert bytes2hex(open(sha256,raw))==before
            @assert !ispath(output)
            open(output,"w") do io
                println(io,"pressure_pa,sg_physical")
                writedlm(io,data,',')
            end
            println(entries,join((slot,entry,Int(!accepted),err),','));flush(entries)
            println("PC_STAGED slot=",slot," reused=",accepted," entry_pa=",entry);flush(stdout)
            GC.gc()
        end
    finally
        close(entries)
    end
end
main()
