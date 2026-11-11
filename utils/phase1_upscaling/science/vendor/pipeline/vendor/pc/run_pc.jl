include(joinpath(@__DIR__,"src","FaultCorePc.jl"))
using .FaultCorePc
length(ARGS)==2 || error("Usage: julia --startup-file=no --math-mode=ieee run_pc.jl raw.bin NEW_OUTPUT_DIRECTORY")
run_file(abspath(ARGS[1]),abspath(ARGS[2]))
