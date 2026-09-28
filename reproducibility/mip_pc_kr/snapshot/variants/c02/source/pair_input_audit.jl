import TOML
import SHA
import ForwardDiff

function verified_angle(x, angle)
    x = x isa AbstractArray ? only(x) : x
    @assert x isa Real && !(x isa Bool) && isfinite(x) && x == angle
end

function audit_candidate_pair(candidate_path, out)
    baseline = MAT.matread(ENV["BASELINE_SPECIFIC_MAT"])
    candidate = MAT.matread(candidate_path)
    angle = parse(Int, ENV["MIP_WRR_ANGLE"])
    convention = ENV["MIP_PC_CONVENTION"]
    @assert angle in (30,60)
    @assert convention in ("physical_prebreakthrough", "zero_before_breakthrough")
    @assert !haskey(baseline["metadata"], "wrr_contact_angle_deg")
    verified_angle(candidate["metadata"]["wrr_contact_angle_deg"], angle)
    @assert candidate["curve_selection"]["pc_convention"] == convention
    @assert candidate["metadata"]["kr_representation"] == "mip_dip_parallel_full_slice"
    allowed = Set([
        "name", "created", "curve_selection", "candidate_export",
        "downstream_contract.validation", "downstream_contract.source_geometry_validation",
        "metadata.pc_representation", "metadata.kr_representation",
        "metadata.representation_detection", "metadata.reservoir_ready_file",
        "metadata.reservoir_ready_schema", "metadata.sgof_point_count_min",
        "metadata.sgof_point_count_max", "metadata.pc_curve_treatment",
        "metadata.note", "metadata.provenance",
        "fault.fluid_tables.SGOF.tables", "fault.fluid_tables.SGOF.point_counts",
        "fault.fluid_tables.SGOF.bulk_sg_max", "fault.fluid_tables.SGOF.effective_swi"
    ])
    changed = String[]
    function compare(a, b, path="")
        isequal(a,b) && return
        if path == "metadata.wrr_contact_angle_deg"
            @assert a === nothing
            verified_angle(b, angle)
            push!(changed, path)
        elseif path in allowed
            push!(changed, path)
        elseif a isa AbstractDict && b isa AbstractDict
            for key in union(keys(a), keys(b))
                p = isempty(path) ? string(key) : path*"."*string(key)
                compare(get(a,key,nothing),get(b,key,nothing),p)
            end
        else
            error("Unapproved production/candidate difference: $path")
        end
    end
    compare(baseline,candidate)
    tables = vec(candidate["fault"]["fluid_tables"]["SGOF"]["tables"])
    reference = MAT.matread(ENV["CANDIDATE_TABLES_MAT"])
    verified_angle(reference["wrrContactAngleDeg"], angle)
    @assert reference["pcConvention"] == convention
    indices = Int.(vec(candidate["fault"]["fluid_tables"]["SGOF"]["source_curve_linear_indices"]))
    @assert length(tables)==522 && sort(indices)==collect(1:522)
    @assert all(isequal(tables[i],vec(reference["exportTables"])[indices[i]]) for i in 1:522)
    promotion = candidate["candidate_export"]["production_promotion"]
    @assert !Bool(promotion isa AbstractArray ? only(promotion) : promotion)
    @assert isequal(candidate["downstream_contract"]["source_geometry_validation"],
                    baseline["downstream_contract"]["validation"])
    if convention == "zero_before_breakthrough"
        physical = MAT.matread(ENV["PHYSICAL_TABLES_MAT"])
        old = [vec(physical["exportTables"])[i] for i in indices]
        @assert all(isequal(a[:,1:3], b[:,1:3]) && isequal(a[end,:], b[end,:]) &&
                    all(b[:,4] .>= a[:,4]) for (a,b) in zip(old,tables))
        oldkr = JutulDarcy.deck_relperm(Dict(),Dict("SGOF"=>old);
            oil=true,water=false,gas=true,satnum=collect(1:522))
        newkr = JutulDarcy.deck_relperm(Dict(),Dict("SGOF"=>tables);
            oil=true,water=false,gas=true,satnum=collect(1:522))
        for (i,t) in enumerate(tables)
            points=sort!(unique(vcat(collect(range(0,1;length=501)),t[:,1],
                (t[1:end-1,1]+t[2:end,1])/2)))
            for s in points
                for (a,b,x) in ((oldkr.krg[i],newkr.krg[i],s),(oldkr.krog[i],newkr.krog[i],1-s))
                    @assert isequal(a(x),b(x)) &&
                        isequal(ForwardDiff.derivative(a,x),ForwardDiff.derivative(b,x))
                end
            end
        end
    end
    open(joinpath(out,"pair_input_audit.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true, "changed_fields"=>sort(changed),
            "all_other_specific_fields_exactly_equal"=>true,
            "all_522_tables_match_validated_export"=>true,
            "wrr_contact_angle_deg"=>angle, "pc_convention"=>convention,
            "pc_pair_kr_nodes_and_derivatives_exact"=>(convention=="zero_before_breakthrough"),
            "common_input_is_identical_checksum_pinned_file"=>true,
            "candidate_sha256"=>bytes2hex(open(SHA.sha256,candidate_path)),
            "baseline_sha256"=>bytes2hex(open(SHA.sha256,ENV["BASELINE_SPECIFIC_MAT"]))))
    end
    println("PAIR_INPUT_AUDIT_PASS: only authorized closure/provenance fields differ")
    flush(stdout)
end
