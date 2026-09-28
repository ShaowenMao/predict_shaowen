using TOML, SHA, LinearAlgebra, DelimitedFiles
const cfg = TOML.parsefile(ARGS[1])
function verify_pins()
    for (path, digest) in cfg["pins"]
        @assert bytes2hex(open(sha256,path)) == digest "Source/input changed: $path"
    end
end
verify_pins()
@assert string(VERSION) == "1.12.6"
include("SelectStates.jl")
using .SelectStates: select_states
include(joinpath(cfg["pipeline"],"FaultCoreUpscaling.jl"))
include(joinpath(cfg["backend"],"Backend.jl"))
include(joinpath(cfg["backend"],"MemoryProbe.jl"))
const U = FaultCoreUpscaling
const Pc = U.Pc
BLAS.set_num_threads(2)
U.Flow.bridge!(cfg["bridge"])
U.Diagnostics.verify_receipt!(U._source_hash_impl,cfg["pipeline_hash"])
U.Diagnostics.reset!(cfg["pipeline_hash"])
include("Kernel.jl")
include("Comparison.jl")
include("comparison_tests.jl")

function saved_numeric(path)
    Matrix{Float64}(readdlm(path,',';skipstart=1))
end

function one_core(c, out)
    input = Pc.read_input(c["raw"])
    g = U.read_geometry(c["geometry"])
    U.validate_geometry(input,g,open(sha256,c["raw"]))
    @assert length(input.pressures)==100
    @assert input.settings.co2_brine_theta_deg==60
    @assert input.settings.sand_endpoint==.95 && input.settings.clay_endpoint==.8
    if haskey(c,"recover_artifacts")
        @assert !ispath(out);mkpath(out)
        for n in ("pc100.csv","kr.csv","selection.csv","summary.toml","flow_calls.jsonl")
            path=joinpath(c["recover_artifacts"],n)
            @assert haskey(cfg["pins"],path)
            @assert bytes2hex(open(sha256,path))==cfg["pins"][path]
            cp(path,joinpath(out,n))
        end
        println("RECOVER_COMPLETED_CURVE ",c["key"]);flush(stdout)
    else
        pc,kr = run_curve(input,g,out,"sparse10")
    end
    refpc = readlines(c["reference_pc"])
    actualpc = readlines(joinpath(out,"pc100.csv"))
    aggregate_difference=compare_pc_lines(actualpc,refpc)
    refkr = saved_numeric(c["reference_kr"])
    x = saved_numeric(joinpath(out,"kr.csv"))
    ids = Int.(x[:,1])
    @assert length(ids)<=10 && issorted(ids) && length(unique(ids))==length(ids)
    @assert ids[1]==1 && ids[end]==100
    refids,_=select_states(refkr[:,3],10)
    @assert ids==refids "Selected existing MIP states changed"
    @assert x[:,1:2]==refkr[ids,1:2]
    @assert all(same_average.(x[:,3],refkr[ids,3]))
    delta = abs.(x[:,4:5].-refkr[ids,4:5])
    @assert maximum(delta)<=1e-8
    @assert all(delta .<= 1e-12 .+ 1e-5.*abs.(refkr[ids,4:5]))
    @assert (x[:,4:5].==0)==(refkr[ids,4:5].==0)
    summary = TOML.parsefile(joinpath(out,"summary.toml"))
    @assert summary["phase_solver_calls"]<=20 && summary["intrinsic_solver_calls"]==1
    life = summary["native_lifetime"]
    @assert life["symbolic_created"]==life["symbolic_released"]
    @assert life["numeric_created"]==life["numeric_released"]
    # Compare the independently inherited exact-entry audit with the actual model.
    model,_ = Pc.prepare_model(input)
    entry = minimum(model.entry[model.inlet])
    audit = TOML.parsefile(c["scenario_audit"])
    @assert entry==audit["entry_pressure_pa"]
    Dict("passed"=>true,"key"=>c["key"],"raw_sha256"=>bytes2hex(open(sha256,c["raw"])),
        "geometry_sha256"=>bytes2hex(open(sha256,c["geometry"])),
        "selected_node_max_abs_difference"=>maximum(delta),
        "saved_state_hashes_and_pc_pressures_equal"=>true,
        "bulk_average_max_abs_difference"=>aggregate_difference,
        "aggregate_roundoff_relative_bound"=>32*eps(Float64),
        "entry_pressure_pa"=>entry,"numerical_reused"=>haskey(c,"recover_artifacts"),
        "factor_lifetime_balanced"=>true,"full100_flow_recomputed"=>false,
        "source_lock_sha256"=>cfg["source_lock_sha256"],"job"=>ENV["SLURM_JOB_ID"],
        "artifacts"=>Dict(n=>bytes2hex(open(sha256,joinpath(out,n))) for n in
            ("pc100.csv","kr.csv","selection.csv","summary.toml","flow_calls.jsonl")))
end

function run_batch()
    out=cfg["output"];@assert !ispath(out);mkpath(out)
    records=[]
    for (i,c) in enumerate(cfg["cases"])
        slot=lpad(string(i),2,'0'); folder=joinpath(out,slot*"_"*c["key"])
        println("CORE_START index=",i," key=",c["key"]);flush(stdout)
        if haskey(c,"reuse_receipt")
            old=TOML.parsefile(c["reuse_receipt"])
            @assert old["passed"] && old["source_lock_sha256"]==cfg["source_lock_sha256"]
            @assert old["key"]==c["key"]
            @assert old["raw_sha256"]==bytes2hex(open(sha256,c["raw"]))
            @assert old["geometry_sha256"]==bytes2hex(open(sha256,c["geometry"]))
            for (n,h) in old["artifacts"]
                @assert bytes2hex(open(sha256,joinpath(dirname(c["reuse_receipt"]),n)))==h
            end
            push!(records,Dict("index"=>i,"key"=>c["key"],"reused"=>true,
                "receipt"=>c["reuse_receipt"],"receipt_sha256"=>bytes2hex(open(sha256,c["reuse_receipt"]))))
            continue
        end
        before=MemoryProbe.after_gc()
        start=time_ns(); receipt=one_core(c,folder); elapsed=(time_ns()-start)/1e9
        after=MemoryProbe.after_gc()
        receipt["memory_after_gc"]=after
        receipt["elapsed_with_review_seconds"]=elapsed
        receipt["index"]=i
        open(joinpath(folder,"CORE_PASS.toml"),"w") do io; TOML.print(io,receipt); end
        push!(records,Dict("index"=>i,"key"=>c["key"],"reused"=>false,
            "receipt"=>joinpath(folder,"CORE_PASS.toml"),
            "receipt_sha256"=>bytes2hex(open(sha256,joinpath(folder,"CORE_PASS.toml"))),
            "before_gc_rss_kib"=>before["VmRSS_kib"],"after_gc_rss_kib"=>after["VmRSS_kib"]))
        println("CORE_PASS index=",i," key=",c["key"]," seconds=",elapsed,
            " rss_kib=",after["VmRSS_kib"]);flush(stdout)
        # Preserve the finished core even when the batch cannot safely continue.
        @assert after["VmRSS_kib"]*1024 <= .90*cfg["memory_mib"]*1024^2 "Retained-memory guard"
        if i==get(cfg,"stop_after",0)
            verify_pins()
            open(joinpath(out,"PREFIX_PASS.toml"),"w") do io
                TOML.print(io,Dict("passed"=>true,"intentional_boundary_stop"=>true,"records"=>records))
            end
            println("INTENTIONAL_PREFIX_STOP ",i);flush(stdout)
            return
        end
    end
    verify_pins()
    open(joinpath(out,"BATCH_PASS.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"records"=>records,"source_lock_sha256"=>cfg["source_lock_sha256"]))
    end
    println("BATCH_PASS ",length(records));flush(stdout)
end

if cfg["mode"]=="bootstrap"
    verify_pins()
    println("BOOTSTRAP_LOAD_PASS ",VERSION);flush(stdout)
else
    @assert cfg["mode"]=="batch"
    run_batch()
end
