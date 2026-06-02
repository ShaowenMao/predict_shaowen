"""
    step01_load_level2_states(config)

Load the six Level 2 window-state objects for one geology.
"""
function step01_load_level2_states(config::Dict{String, Any})
    return Level3IO.load_level2_states(config["level2_state_root"],
                                       String.(config["fixed_windows"]))
end

