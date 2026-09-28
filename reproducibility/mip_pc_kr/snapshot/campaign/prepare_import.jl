using TOML
c=TOML.parsefile(ARGS[1])
@assert string(VERSION)=="1.12.6"
ENV["FROZEN_SOURCE"]=c["frozen"]
include(c["scenario"])
println("FROZEN_PREPARATION_IMPORT_PASS")
