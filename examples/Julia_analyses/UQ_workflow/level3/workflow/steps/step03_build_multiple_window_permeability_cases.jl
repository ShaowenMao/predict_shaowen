"""
    step03_build_multiple_window_permeability_cases(grouping_result, distance_result, config)

Build the 10-case Level 3 multiple-window permeability case set from final
window similarity groups.

This step enumerates binary group splits, scores each split, selects the best
grouped low/high pattern, builds the four grouped low/high cases, and assembles
the final 10 multiple-window permeability cases.
"""
function step03_build_multiple_window_permeability_cases(grouping_result::Dict{String, Any},
                                                         distance_result::Dict{String, Any},
                                                         config::Dict{String, Any})
    return Level3PermeabilityCases.build_multiple_window_permeability_cases(
        grouping_result,
        distance_result;
        geology_id = String(config["geology_id"]),
    )
end
