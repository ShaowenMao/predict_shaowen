module UQWorkflow

include(joinpath(@__DIR__, "..", "sampling", "lib", "IndependentFullFaultSampling.jl"))

using .IndependentFullFaultSampling

export IndependentFullFaultSampling

end
