import TOML, SHA
function audit_candidate_pair(path, out)
    baseline = ENV["BASELINE_SPECIFIC_MAT"]
    digest = bytes2hex(open(SHA.sha256, path))
    @assert digest == bytes2hex(open(SHA.sha256, baseline))
    open(joinpath(out,"pair_input_audit.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"all_specific_fields_exactly_equal"=>true,
            "production_properties_unchanged"=>true,"candidate_sha256"=>digest,
            "baseline_sha256"=>digest))
    end
end
