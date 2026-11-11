module BatchGuard
using TOML, SHA
const GiB = 1024^3

function write_new(path, value)
    ispath(path) && error("Receipt already exists: " * path)
    temp = path * ".partial"
    ispath(temp) && error("Unresolved partial receipt: " * temp)
    open(temp, "w") do io
        TOML.print(io, value; sorted=true)
        flush(io)
    end
    mv(temp, path; force=false)
end

function reading(root)
    if isfile(joinpath(root, "memory.peak"))
        events = Dict(split(line)[1] => parse(Int, split(line)[2])
                      for line in eachline(joinpath(root, "memory.events")))
        return Dict("peak_bytes"=>parse(Int, read(joinpath(root, "memory.peak"), String)),
                    "limit_bytes"=>parse(Int, read(joinpath(root, "memory.max"), String)),
                    "events"=>sum(get(events, k, 0) for k in ("max", "oom", "oom_kill")),
                    "version"=>2)
    end
    Dict("peak_bytes"=>parse(Int, read(joinpath(root, "memory.max_usage_in_bytes"), String)),
         "limit_bytes"=>parse(Int, read(joinpath(root, "memory.limit_in_bytes"), String)),
         "events"=>parse(Int, read(joinpath(root, "memory.failcnt"), String)), "version"=>1)
end

function memory_pass(m, expected_gib)
    all(haskey(m, k) for k in ("peak_bytes", "limit_bytes", "events")) || return false
    peak, limit = m["peak_bytes"], m["limit_bytes"]
    peak >= 0 && limit == expected_gib * GiB && m["events"] == 0 &&
        max(peak + GiB, 1.3 * peak) <= limit
end

struct Session
    config::Dict{String,Any}
    guards::Dict{String,Any}
    started::UInt64
end

function Session(config::Dict{String,Any})
    Sys.islinux() || error("This worker requires Engaging Linux qualification")
    string(VERSION) == "1.12.6" || error("Julia version mismatch")
    g = config["batch_guard"]
    g["qualification_only"] === true || error("Not approved for campaign execution")
    get(ENV, "SLURM_JOB_ID", "") == g["job_id"] || error("Wrong Slurm job")
    root = realpath(g["cgroup"])
    startswith(root, "/sys/fs/cgroup/") || error("Not a cgroup path")
    ("job_" * g["job_id"]) in splitpath(root) || error("Not the whole-job cgroup")
    length(config["cases"]) in (1, 2, 4, 8) || error("Unsupported qualification length")
    keys = [c["numerical_key"] for c in config["cases"]]
    if get(g, "retention_probe", false)
        length(keys) in (4, 8) && first(keys) == last(keys) && length(unique(keys)) >= 2 ||
            error("Invalid sentinel retention pilot")
    else
        length(unique(keys)) == length(keys) || error("Duplicate owners must be resolved upstream")
    end
    g["deadline_unix"] > time() || error("Deadline already passed")
    g["retained_rss_limit_gib"] > 0 || error("Missing retained-RSS pilot guard")
    Session(config, g, time_ns())
end

function before!(s, pass)
    g, c = s.guards, s.config["cases"][pass]
    time() + c["seconds_upper"] + g["shutdown_margin_seconds"] < g["deadline_unix"] ||
        error("STOP_BEFORE_NEXT_CORE: insufficient wall time")
    memory_pass(reading(g["cgroup"]), g["memory_gib"]) ||
        error("STOP_BEFORE_NEXT_CORE: memory events or headroom")
    println("BATCH_CORE_START pass=", pass, " numerical_key=", c["numerical_key"],
            " since_worker_seconds=", (time_ns()-s.started)/1e9)
    flush(stdout)
end

function artifact_map(folder)
    files = Dict{String,String}()
    for (root, dirs, names) in walkdir(folder)
        any(islink(joinpath(root, n)) for n in vcat(dirs, names)) && error("Symlink in results")
        for n in names
            p = joinpath(root, n)
            files[relpath(p, folder)] = bytes2hex(open(sha256, p))
        end
    end
    files
end

function record!(s, pass, after_gc)
    config, g = s.config, s.guards
    folder = joinpath(config["output"], "pass" * lpad(string(pass), 2, '0'))
    completed = TOML.parsefile(joinpath(folder, "completion.toml"))
    completed["execution_complete"] === true && completed["points"] == 100 ||
        error("Incomplete numerical curve")
    files = artifact_map(folder)
    for (name, expected) in completed["outputs"]
        get(files, name, "") == expected || error("Completion artifact mismatch: " * name)
    end
    m = reading(g["cgroup"])
    retained = after_gc["VmRSS_kib"] * 1024 / GiB
    resource_ok = memory_pass(m, g["memory_gib"]) && retained <= g["retained_rss_limit_gib"]
    timing = joinpath(config["output"], "pass$(pass)_timing.toml")
    receipt = Dict("status"=>"NUMERICAL_COMPLETE_REVIEW_REQUIRED", "pass"=>pass,
        "numerical_key"=>config["cases"][pass]["numerical_key"],
        "raw_hash"=>config["cases"][pass]["raw_hash"],
        "geometry_hash"=>config["cases"][pass]["geometry_hash"],
        "pipeline_hash"=>config["pipeline_hash"], "release_id"=>g["release_id"],
        "config_sha256"=>g["config_sha256"], "artifacts"=>files,
        "timing_sha256"=>bytes2hex(open(sha256, timing)),
        "whole_job_memory"=>m, "retained_after_core"=>after_gc,
        "resource_passed"=>resource_ok, "scientific_comparison_passed"=>false,
        "native_import_passed"=>false, "campaign_accepted"=>false)
    write_new(joinpath(config["output"], "pass$(pass)_handoff.toml"), receipt)
    println("BATCH_CORE_PRESERVED pass=", pass, " resource_passed=", resource_ok)
    flush(stdout)
    resource_ok || error("STOP_AFTER_PRESERVED_CORE: resource review required")
end

function finish!(s)
    write_new(joinpath(s.config["output"], "BATCH_EXECUTION_COMPLETE.toml"),
        Dict("passes"=>s.config["passes"], "qualification_only"=>true,
             "campaign_accepted"=>false, "seconds"=>(time_ns()-s.started)/1e9))
end
end
