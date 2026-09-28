using SHA, TOML
include(joinpath(@__DIR__, "vendor_pc/src/FaultCorePc.jl"))
const P = FaultCorePc
length(ARGS) == 3 || error("manifest_tsv entry_csv output_directory")
manifest, entry_csv, out = ARGS
@assert haskey(ENV, "SLURM_JOB_ID") "Compute-node execution required"
@assert !ispath(out)
mkpath(out)
entries = Dict(parse(Int, split(l, ',')[1]) => parse(Float64, split(l, ',')[2])
    for l in readlines(entry_csv)[2:end])
lines = readlines(manifest)[2:end]
@assert length(lines) == length(entries) == 522
seen = Set{Int}()
started = time()
open(joinpath(out, "breakthrough.csv"), "w") do io
    println(io, "slot,entry_pressure_pa,breakthrough_pa,minimum_mobility_loss_pressure_pa,input_sha256")
    for line in lines
        ss, path, hash = split(line, '\t')
        slot = parse(Int, ss)
        @assert !(slot in seen)
        push!(seen, slot)
        @assert bytes2hex(open(sha256, path)) == hash
        x = P.read_input(path)
        m, _ = P.prepare_model(x)
        @assert x.core == slot
        pe = minimum(m.entry[m.inlet])
        @assert pe == entries[slot] "Local scaling differs from validated candidate"
        @assert all(c -> all(c.brine[1:end-1] .> 0) && c.brine[end] == 0, m.curves)
        wet = zeros(UInt8, m.n)
        P.StandaloneMIP.reach!(wet, m.row, m.neighbor, ones(UInt8, m.n), m.outlet, zeros(Int, m.n))
        @assert all(wet .!= 0) "Initially isolated wetting volume"
        arrival = P.path_thresholds(m, m.entry, m.inlet; gas=true)
        pb = minimum(arrival[m.outlet])
        # Before any cell loses brine mobility, this minimax threshold is also
        # the exact breakthrough threshold of the outlet-trapping algorithm.
        @assert isfinite(pb) && pe <= pb < minimum(m.endpoint) "Trapping-aware event recovery required"
        @assert m.pressures[1] <= pb <= m.pressures[end]
        println(io, join((slot, repr(pe), repr(pb), repr(minimum(m.endpoint)), hash), ','))
        flush(io)
        slot % 25 == 0 && (println("BREAKTHROUGH ", slot, "/522"); flush(stdout))
        GC.gc()
    end
end
@assert seen == Set(1:522)
open(joinpath(out, "PASS.toml"), "w") do io
    TOML.print(io, Dict("passed" => true, "cores" => 522,
        "method" => "exact_minimax_inlet_to_outlet_before_any_wetting_mobility_loss",
        "original_entry_pressures_exactly_reproduced" => true,
        "mip_sweeps" => 0, "phase_flow_solves" => 0,
        "seconds" => time()-started, "julia_version" => string(VERSION),
        "manifest_sha256" => bytes2hex(open(sha256, manifest)),
        "entry_metadata_sha256" => bytes2hex(open(sha256, entry_csv)),
        "breakthrough_sha256" => bytes2hex(open(sha256, joinpath(out,"breakthrough.csv")))))
end
