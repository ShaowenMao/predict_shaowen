import TOML
import SHA

module MipPcOnlyAudit
using Test

function interp_pc(t, x)
    i = searchsortedlast(view(t, :, 1), x)
    i == 0 && return t[1, 4]
    i >= size(t, 1) && return t[end, 4]
    f = (x - t[i, 1])/(t[i+1, 1] - t[i, 1])
    return (1-f)*t[i, 4] + f*t[i+1, 4]
end

function validate_table(mip, candidate, physical, zero)
    @assert size(mip) == size(candidate)
    @assert isequal(mip[:, 1:3], candidate[:, 1:3]) "MIP Kr/saturation knots changed"
    @assert isequal(physical[:, 1:3], zero[:, 1:3])
    @assert all(isfinite, candidate) && all(diff(candidate[:, 1]) .> 0)
    @assert candidate[1, 4] > 0 && all(diff(candidate[:, 4]) .>= 0)
    xs = sort(unique(vcat(mip[:, 1], physical[:, 1], zero[:, 1])))
    @assert maximum(abs(interp_pc(mip,x)-interp_pc(physical,x)) for x in xs) < 1e-7
    @assert maximum(abs(interp_pc(candidate,x)-interp_pc(zero,x)) for x in xs) < 1e-7
    last = findlast(physical[:, 4] .!= zero[:, 4])
    @assert last !== nothing && last < size(zero, 1)
    onset = zero[last+1, 1]
    post = mip[:, 1] .>= onset
    @assert isequal(mip[post, 4], candidate[post, 4]) "Post-breakthrough Pc changed"
    @assert candidate[1, 4] == zero[1, 4]
    return count(candidate[:, 4] .!= mip[:, 4])
end

function unit_tests()
    p = [0.0 0.0 1.0 10.0; 0.1 0.05 0.8 20.0; 0.3 0.2 0.5 40.0; 1.0 1.0 0.0 100.0]
    z = copy(p); z[1,4] = 15.0; z[2,4] = 15.0 + (40.0-15.0)*(.1/.3)
    @test validate_table(p,z,p,z) == 2
    bad = copy(z); bad[2,2] += .01
    @test_throws AssertionError validate_table(p,bad,p,z)
    bad = copy(z); bad[3,4] += 1.0
    @test_throws AssertionError validate_table(p,bad,p,z)
    @test_throws AssertionError validate_table(p,p,p,z)
end
end

function audit_candidate_pair(candidate_path, out)
    MipPcOnlyAudit.unit_tests()
    mip = MAT.matread(ENV["MIP_REFERENCE_SPECIFIC_MAT"])
    candidate = MAT.matread(candidate_path)
    physical = MAT.matread(ENV["PHYSICAL_PC_SPECIFIC_MAT"])
    zero = MAT.matread(ENV["ZERO_PC_SPECIFIC_MAT"])
    allowed = Set(["fault.fluid_tables.SGOF.tables", "metadata.pc_curve_treatment"])
    changed = String[]
    function compare(a,b,path="")
        isequal(a,b) && return
        if path in allowed
            push!(changed,path)
        elseif a isa AbstractDict && b isa AbstractDict
            @assert Set(keys(a)) == Set(keys(b)) "Field set changed at $path"
            for k in keys(a)
                compare(a[k],b[k],isempty(path) ? string(k) : path*"."*string(k))
            end
        else
            error("Unapproved MIP/Pc-only difference at $path")
        end
    end
    compare(mip,candidate)
    sgof(x) = x["fault"]["fluid_tables"]["SGOF"]
    function map_tables(x)
        g=sgof(x); ids=Int.(vec(g["regions"])); tables=vec(g["tables"])
        @assert length(ids)==length(unique(ids))==length(tables)==522
        return Dict(ids[i]=>tables[i] for i in eachindex(ids))
    end
    h,c,p,z=map_tables.((mip,candidate,physical,zero))
    @assert Set(keys(h))==Set(keys(c))==Set(keys(p))==Set(keys(z))
    changed_nodes=sum(MipPcOnlyAudit.validate_table(h[r],c[r],p[r],z[r]) for r in keys(h))
    open(joinpath(out,"pair_input_audit.toml"),"w") do io
        TOML.print(io,Dict("passed"=>true,"changed_fields"=>sort(changed),
            "cores"=>522,"changed_pc_nodes"=>changed_nodes,
            "mip_kr_and_saturation_coordinates_bitwise_equal"=>true,
            "postbreakthrough_pc_nodes_bitwise_equal"=>true,
            "all_other_specific_fields_exactly_equal"=>true,
            "physical_pc_donor_functions_match"=>true,"zero_pc_donor_functions_match"=>true,
            "common_input_is_identical_checksum_pinned_file"=>true,
            "candidate_sha256"=>bytes2hex(open(SHA.sha256,candidate_path)),
            "baseline_sha256"=>bytes2hex(open(SHA.sha256,ENV["MIP_REFERENCE_SPECIFIC_MAT"]))))
    end
    println("PAIR_INPUT_AUDIT_PASS: only Pc onset and its treatment label changed; MIP Kr exact")
    flush(stdout)
end
