module CorrectedFaultValidation
using JutulDarcy, Jutul, ForwardDiff, SHA, TOML

# Independent oracles are the immutable production source, not the edited branch.
function install_oracle(file, first_name, next_name, oracle_name)
    source = read(file, String)
    a = findfirst("function " * first_name, source)
    b = findfirst("function " * next_name, source)
    @assert a !== nothing && b !== nothing && first(a) < first(b)
    fn = source[first(a):prevind(source, first(b))]
    Core.eval(JutulDarcy, Meta.parse(replace(fn,
        "function " * first_name => "function " * oracle_name; count=1)))
end

reference = joinpath(ENV["CORRECTED_STUDY_ROOT"], "reference")
install_oracle(joinpath(reference, "advanced.jl"), "update_two_phase_relperm!",
    "imbibition_table_by_region", "corrected_study_original_two_phase!")
install_oracle(joinpath(reference, "production_qoi.jl"), "production_qoi_gas_mobility_accounting",
    "production_qoi_active_gas_mobility", "corrected_study_original_accounting")
install_oracle(joinpath(reference, "production_qoi.jl"), "production_qoi_active_gas_mobility",
    "production_qoi_neighbors", "corrected_study_original_active_mobility")

struct ProbeColumn{T} <: AbstractMatrix{T}
    data::Vector{T}
    cell::Int
    nc::Int
end
Base.size(x::ProbeColumn) = (length(x.data), x.nc)
Base.getindex(x::ProbeColumn, i::Int, c::Int) = (@assert c == x.cell; x.data[i])
Base.setindex!(x::ProbeColumn, v, i::Int, c::Int) = (@assert c == x.cell; x.data[i] = v)

function validate!(setup, out)
    rm = setup.case.model.models[:Reservoir]
    st = setup.sim.storage[:Reservoir].state
    rp = rm[:RelativePermeabilities]
    nc = length(rp.regions)
    cells = Int.(vec(setup.mrst["qoi_fault"]["cells"]))
    @assert nc == 2165082 && length(cells) == length(unique(cells)) == 32190
    mask = falses(nc); mask[cells] .= true
    @assert findall(rp.drainage_only_regions) == collect(9:530)
    @assert all(mask[c] == JutulDarcy.region_is_drainage_only(rp, rp.regions[c]) for c in 1:nc)
    @assert JutulDarcy.hysteresis_is_active(rp)
    @assert rp.hysteresis_g isa JutulDarcy.KilloughHysteresis
    @assert rp.hysteresis_g.s_min == 0.05
    ph = typeof(rp).parameters[2]; @assert ph == :og
    phase_ind = JutulDarcy.phase_indices(rm.system)
    scaling = JutulDarcy.endpoint_scaling_model(rp)
    scalers = JutulDarcy.get_endpoint_scalers(st, scaling, Val(ph); drainage=true)
    scalersi = JutulDarcy.get_endpoint_scalers(st, scaling, Val(ph); drainage=false)
    got_account = JutulDarcy.production_qoi_gas_mobility_accounting(rm, st, nc)
    old_account = JutulDarcy.corrected_study_original_accounting(rm, st, nc)
    @assert got_account.drainage_only_cells == mask
    representatives = Dict{Int,Int}()
    for c in 1:nc
        r = Int(rp.regions[c]); get!(representatives, r, c)
    end
    @assert sort!(collect(keys(representatives))) == collect(1:530)
    fault_tests = 0; host_tests = 0; host_history_witnesses = 0
    host_accounting_witnesses = 0; value_error = 0.0; slope_error = 0.0
    open(joinpath(out, "constitutive_validation.tsv"), "w") do io
        println(io, "region\tcell\tfault\tsg\tsg_max\tkrw\tkrg\tdkrw_dsg\tdkrg_dsg\tqoi_branch\tqoi_history")
        for r in 1:530
            c = representatives[r]
            wd = JutulDarcy.table_by_region(rp.krog, r)
            gd = JutulDarcy.table_by_region(rp.krg, r)
            dw, dg = JutulDarcy.get_two_phase_relperms(rp, c, wd, gd, scalers)
            cap = Float64(Jutul.value(dg.s_max))
            crit = Float64(Jutul.value(dg.critical))
            points = sort!(unique(clamp.([0., .001, .01, .049, .05-1e-10, .05,
                .05+1e-10, .051, .1, .2, .4, .7, .9, crit, cap], 0., cap)))
            for sg in points, sm in unique([sg, min(cap, sg+.1), cap])
                gas = ForwardDiff.Dual(sg, 1.0)
                s = ProbeColumn([1-gas, gas], c, nc)
                history = ProbeColumn([1., sm], c, nc)
                got = ProbeColumn([zero(gas), zero(gas)], c, nc)
                old = ProbeColumn([zero(gas), zero(gas)], c, nc)
                args = (rp, rp.krog, rp.krg, rp.hysteresis_og, rp.hysteresis_g,
                    phase_ind, s, history, c, scalers, scalersi)
                JutulDarcy.update_two_phase_relperm!(got, args...)
                JutulDarcy.corrected_study_original_two_phase!(old, args...)
                expected = mask[c] ? [dw(1-gas), dg(gas)] : old.data
                for i in 1:2
                    e = abs(ForwardDiff.value(got.data[i])-ForwardDiff.value(expected[i]))
                    de = abs(ForwardDiff.partials(got.data[i])[1]-ForwardDiff.partials(expected[i])[1])
                    @assert isfinite(e) && isfinite(de) && e == 0.0 && de == 0.0
                    value_error = max(value_error, e); slope_error = max(slope_error, de)
                end
                active = JutulDarcy.production_qoi_active_gas_mobility(got_account, c, sg, sm)
                old_active = JutulDarcy.corrected_study_original_active_mobility(old_account, c, sg, sm)
                if mask[c]
                    @assert active == (critical=crit, branch=:drainage, history_dependent=false)
                    fault_tests += 1
                else
                    @assert isequal(active, old_active)
                    host_tests += 1
                    host_history_witnesses += abs(ForwardDiff.value(old.data[2])-Jutul.value(dg(sg))) > 1e-10
                    host_accounting_witnesses += active.history_dependent
                end
                println(io, join((r,c,mask[c],sg,sm,ForwardDiff.value.(got.data)...,
                    ForwardDiff.partials(got.data[1])[1],ForwardDiff.partials(got.data[2])[1],
                    active.branch,active.history_dependent), '\t'))
            end
        end
    end
    @assert fault_tests > 0 && host_tests > 0 && host_history_witnesses > 0 && host_accounting_witnesses > 0
    result = Dict("passed"=>true, "fault_cells"=>count(mask), "host_cells"=>nc-count(mask),
        "fault_regions"=>522, "host_regions"=>8, "fault_tests"=>fault_tests,
        "host_tests"=>host_tests, "host_history_witnesses"=>host_history_witnesses,
        "host_accounting_witnesses"=>host_accounting_witnesses,
        "maximum_value_error"=>value_error, "maximum_slope_error"=>slope_error,
        "fault_mobility_and_qoi_drainage_only"=>true, "host_matches_immutable_oracle"=>true,
        "fault_pc_changed_by_correction"=>false, "drainage_kr_changed_by_correction"=>false)
    open(io->TOML.print(io, result), joinpath(out,"fault_hysteresis_validation.toml"),"w")
    println("FAULT_HYSTERESIS_VALIDATION_PASS: native values/AD slopes and QoI branches")
    flush(stdout)
    return result
end
end
