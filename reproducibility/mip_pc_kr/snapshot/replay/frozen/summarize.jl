using TOML
include("vendor/pipeline/vendor/kr/json_report.jl")
@assert length(ARGS)==2&&!ispath(ARGS[2])
open(ARGS[2],"w") do io;json_value(io,TOML.parsefile(ARGS[1]));println(io);end
