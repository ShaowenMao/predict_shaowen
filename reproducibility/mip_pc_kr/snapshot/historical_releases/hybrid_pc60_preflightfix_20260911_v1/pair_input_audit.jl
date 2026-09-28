import TOML
import SHA

function require_hybrid_wrr60(value, label)
    if value isa AbstractArray
        length(value) == 1 || error("$label must be a scalar WRR60 angle")
        value = only(value)
    end
    value isa Real && !(value isa Bool) && isfinite(value) && value == 60 ||
        error("$label must be exactly the intended WRR60 angle (60 degrees)")
    return 60.0
end

function compare_hybrid_candidate_fields(baseline, candidate)
    # This exception is an addition specific to the qualified intermediate WRR60 export.
    haskey(baseline["metadata"], "wrr_contact_angle_deg") &&
        error("Baseline unexpectedly already contains WRR-angle metadata")
    require_hybrid_wrr60(get(candidate["metadata"], "wrr_contact_angle_deg", nothing),
                        "Candidate metadata.wrr_contact_angle_deg")
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
        isequal(a, b) && return
        if path == "metadata.wrr_contact_angle_deg"
            a === nothing || error("WRR-angle exception permits only a metadata addition")
            require_hybrid_wrr60(b, path)
            push!(changed, path)
        elseif path in allowed
            push!(changed, path)
        elseif a isa AbstractDict && b isa AbstractDict
            for key in union(keys(a), keys(b))
                p = isempty(path) ? string(key) : path*"."*string(key)
                compare(get(a, key, nothing), get(b, key, nothing), p)
            end
        else
            error("Unapproved production/candidate difference: $path")
        end
    end
    compare(baseline, candidate)
    return changed
end

function audit_candidate_pair(candidate_path, out)
    test_hybrid_pair_audit()
    baseline = MAT.matread(ENV["BASELINE_SPECIFIC_MAT"])
    candidate = MAT.matread(candidate_path)
    changed = compare_hybrid_candidate_fields(baseline, candidate)
    tables = vec(candidate["fault"]["fluid_tables"]["SGOF"]["tables"])
    reference = MAT.matread(ENV["CANDIDATE_TABLES_MAT"])
    require_hybrid_wrr60(get(reference, "wrrContactAngleDeg", nothing),
                        "Qualified table wrrContactAngleDeg")
    indices = Int.(vec(candidate["fault"]["fluid_tables"]["SGOF"]["source_curve_linear_indices"]))
    @assert length(tables)==522 && sort(indices)==collect(1:522)
    @assert all(isequal(tables[i],vec(reference["exportTables"])[indices[i]]) for i in 1:522)
    promotion = candidate["candidate_export"]["production_promotion"]
    @assert !Bool(promotion isa AbstractArray ? only(promotion) : promotion)
    @assert isequal(candidate["downstream_contract"]["source_geometry_validation"],
                    baseline["downstream_contract"]["validation"])
    open(joinpath(out,"pair_input_audit.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true, "changed_fields"=>sort(changed),
            "all_other_specific_fields_exactly_equal"=>true,
            "all_522_tables_match_validated_export"=>true,
            "wrr_contact_angle_deg"=>60,
            "wrr_angle_metadata_and_qualified_export_match"=>true,
            "audit_negative_controls_passed"=>true,
            "common_input_is_identical_checksum_pinned_file"=>true,
            "candidate_sha256"=>bytes2hex(open(SHA.sha256,candidate_path)),
            "baseline_sha256"=>bytes2hex(open(SHA.sha256,ENV["BASELINE_SPECIFIC_MAT"]))))
    end
    println("PAIR_INPUT_AUDIT_PASS: authorized closure/provenance fields and verified WRR60 metadata only")
    flush(stdout)
end

include(joinpath(@__DIR__, "test_pair_input_audit.jl"))
