using Test

function test_hybrid_pair_audit()
    @testset "Isolated WRR60 audit: preserve protected inputs" begin
        baseline = Dict{String,Any}(
            "metadata"=>Dict{String,Any}("geology"=>"same draw"),
            "rock"=>Dict{String,Any}("K"=>[1.0, 2.0], "phi"=>[0.2, 0.3]),
            "fault"=>Dict{String,Any}("fluid_tables"=>Dict{String,Any}(
                "SGOF"=>Dict{String,Any}("tables"=>[1.0], "source_curve_linear_indices"=>[1, 2]))))
        candidate = deepcopy(baseline)
        candidate["metadata"]["wrr_contact_angle_deg"] = 60.0
        @test compare_hybrid_candidate_fields(baseline, candidate) == ["metadata.wrr_contact_angle_deg"]
        for angle in (60, 60.0, [60.0], reshape([60.0], 1, 1))
            @test require_hybrid_wrr60(angle, "test") == 60.0
        end
        for angle in (30.0, 70.0, NaN, Inf, "60", true, nothing, [], [60.0, 60.0])
            bad = deepcopy(candidate)
            bad["metadata"]["wrr_contact_angle_deg"] = angle
            @test_throws ErrorException compare_hybrid_candidate_fields(baseline, bad)
            @test_throws ErrorException require_hybrid_wrr60(angle, "qualified table")
        end
        @test_throws ErrorException compare_hybrid_candidate_fields(baseline, baseline)
        @test_throws ErrorException compare_hybrid_candidate_fields(candidate, candidate)
        for field in ("K", "phi")
            bad = deepcopy(candidate)
            bad["rock"][field][1] += 0.01
            @test_throws ErrorException compare_hybrid_candidate_fields(baseline, bad)
        end
        for field in ("geology", "unrelated_new_field")
            bad = deepcopy(candidate)
            bad["metadata"][field] = "different"
            @test_throws ErrorException compare_hybrid_candidate_fields(baseline, bad)
        end
        bad = deepcopy(candidate)
        bad["fault"]["fluid_tables"]["SGOF"]["source_curve_linear_indices"] = [2, 1]
        @test_throws ErrorException compare_hybrid_candidate_fields(baseline, bad)
        allowed = deepcopy(candidate)
        allowed["fault"]["fluid_tables"]["SGOF"]["tables"] = [2.0]
        @test Set(compare_hybrid_candidate_fields(baseline, allowed)) ==
            Set(["metadata.wrr_contact_angle_deg", "fault.fluid_tables.SGOF.tables"])
    end
end
