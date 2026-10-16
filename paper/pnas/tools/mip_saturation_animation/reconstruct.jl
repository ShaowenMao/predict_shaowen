using SHA, TOML

const REF = joinpath(@__DIR__, "reference")
include(joinpath(REF, "src", "FaultCorePc.jl"))
const Pc = FaultCorePc

function main()
    length(ARGS) == 3 || error("Usage: reconstruct.jl RAW EXPECTED_SHA256 OUTPUT")
    input_path, expected, out = ARGS
    bytes2hex(open(sha256, input_path)) == expected || error("Raw input hash mismatch")
    mkpath(out)
    for name in ("saturation_states.bin", "reconstructed_history.csv", "reconstruction.toml")
        ispath(joinpath(out, name)) && error("Refusing to overwrite existing reconstruction: " * name)
    end
    input = Pc.read_input(input_path)
    m, audit = Pc.prepare_model(input)
    paths = Pc.prepare_paths(m)
    s, w = Pc.initial(m.n), Pc.Workspace(m.n)
    open(joinpath(out, "saturation_states.bin"), "w") do states
        write(states, Int64[m.n, length(m.pressures)])
        open(joinpath(out, "reconstructed_history.csv"), "w") do history
            println(history, "sample,pressure_pa,sg_physical,trapped_pv_fraction,immobile_pv_fraction,gas_breakthrough,invaded_pv_fraction,events")
            for (j, p) in enumerate(m.pressures)
                events = Pc.advance_pc!(m, s, p, w, paths)
                row = Pc.observe(m, s, w)
                write(states, s.sg)
                println(history, join([j, row.pressure_pa, row.sg_physical, row.trapped_pv_fraction,
                    row.immobile_pv_fraction, Int(row.gas_breakthrough), row.invaded_pv_fraction, events], ','))
            end
        end
    end
    sources = ["src/FaultCorePc.jl", "src/paths.jl", "src/outputs.jl", "vendor/StandaloneMIP.jl"]
    metadata = Dict("input_sha256" => expected, "core" => input.core,
        "julia_version" => string(VERSION), "model_audit" => audit,
        "reconstructed_not_archived_cell_states" => true,
        "new_pressure_flow_solves" => 0, "production_files_changed" => false,
        "path_backend" => (paths === nothing ? "literal" : "exact_thresholds"),
        "sources" => Dict(p => bytes2hex(open(sha256, joinpath(REF, p))) for p in sources))
    open(joinpath(out, "reconstruction.toml"), "w") do io
        TOML.print(io, metadata)
    end
    println("Reconstructed ", length(m.pressures), " states of ", m.n, " cells; no flow solves.")
end

main()
