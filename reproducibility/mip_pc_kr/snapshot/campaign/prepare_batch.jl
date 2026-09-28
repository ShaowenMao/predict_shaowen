using TOML,SHA
c=TOML.parsefile(ARGS[1]);@assert string(VERSION)=="1.12.6"
ENV["FROZEN_SOURCE"]=c["frozen"]
include(c["scenario"])
for r in c["cases"]
    @assert bytes2hex(open(sha256,r["raw"]))==r["raw_hash"]
    @assert bytes2hex(open(sha256,r["geometry"]))==r["geometry_hash"]
    plan(r["raw"],r["geometry"],r["out30"])
    transform(joinpath(r["out30"],"raw.bin"),joinpath(r["out30"],"geometry.bin"),r["out60"])
    println("WRR60_PREPARED ",r["key"]);flush(stdout)
end

