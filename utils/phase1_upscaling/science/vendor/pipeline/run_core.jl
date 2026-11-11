include("FaultCoreUpscaling.jl")
using .FaultCoreUpscaling
@assert length(ARGS) in (4,5) "raw.bin geometry.bin precision_library NEW_OUTPUT [--states]"
@assert length(ARGS)==4 || ARGS[5]=="--states"
run_files(ARGS[1],ARGS[2],ARGS[3],ARGS[4];save_states=length(ARGS)==5)
