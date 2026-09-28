using TOML, SHA
c=TOML.parsefile(ARGS[1])
ENV["FROZEN_SOURCE"]=c["frozen"]
include(c["scenario"])
@assert string(VERSION)=="1.12.6"
@assert bytes2hex(open(sha256,c["raw"]))==c["raw_hash"]
@assert bytes2hex(open(sha256,c["geometry"]))==c["geometry_hash"]
plan(c["raw"],c["geometry"],c["out30"])
transform(joinpath(c["out30"],"raw.bin"),joinpath(c["out30"],"geometry.bin"),c["out60"])
println("PREPARATION_AND_WRR60_TRANSFORM_PASS");flush(stdout)
