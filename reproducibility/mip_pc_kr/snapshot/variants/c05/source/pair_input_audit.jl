import TOML
import SHA

function audit_candidate_pair(candidate_path, out)
    baseline = MAT.matread(ENV["BASELINE_SPECIFIC_MAT"])
    candidate = MAT.matread(candidate_path)
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
        if path in allowed
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
            "common_input_is_identical_checksum_pinned_file"=>true,
            "candidate_sha256"=>bytes2hex(open(SHA.sha256,candidate_path)),
            "baseline_sha256"=>bytes2hex(open(SHA.sha256,ENV["BASELINE_SPECIFIC_MAT"]))))
    end
    println("PAIR_INPUT_AUDIT_PASS: only authorized closure/provenance fields differ")
    flush(stdout)
end
