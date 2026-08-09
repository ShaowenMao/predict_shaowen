"""Return the lowercase SHA-256 digest for a file."""
function file_sha256(path::AbstractString)
    isfile(path) || error("Cannot hash missing file: $path")
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

"""Encode fields unambiguously using byte-length prefixes."""
function length_prefixed_payload(fields...)
    io = IOBuffer()
    for field in fields
        value = string(field)
        write(io, string(ncodeunits(value)))
        write(io, ':')
        write(io, value)
        write(io, '|')
    end
    return take!(io)
end

"""Return a stable SHA-256 digest for an ordered field tuple."""
stable_hash_hex(fields...) = bytes2hex(sha256(length_prefixed_payload(fields...)))

"""
    stable_case_seed(base_seed, geology_id, phase, case_type, replicate_id, window_id; version)

Derive a platform-independent positive 63-bit seed from case identity. The
conversion is explicitly big-endian and therefore does not depend on host byte
order. Restricting the value to 63 bits keeps it lossless in signed Int64 CSV,
MAT, Python, and MATLAB consumers while avoiding practical stream collisions.
"""
function stable_case_seed(base_seed::Integer,
                          geology_id::AbstractString,
                          phase::AbstractString,
                          case_type::AbstractString,
                          replicate_id::Integer,
                          window_id::AbstractString;
                          version::AbstractString = SEED_METHOD_VERSION)
    digest = sha256(length_prefixed_payload(version, base_seed, geology_id, phase,
                                            case_type, replicate_id, window_id))
    seed = zero(UInt64)
    for byte in digest[1:8]
        seed = (seed << 8) | UInt64(byte)
    end
    seed &= UInt64(typemax(Int64))
    return seed == 0 ? UInt64(1) : seed
end

"""
    stable_uniform_index(seed, draw_index, n; version)

Map a deterministic counter to an exactly uniform integer in `1:n` using a
SHA-256 counter stream and rejection sampling. This avoids dependence on
Julia's evolving pseudorandom-number-generator implementation.
"""
function stable_uniform_index(seed::UInt64,
                              draw_index::Integer,
                              n::Integer;
                              version::AbstractString = SEED_METHOD_VERSION)
    n > 0 || error("Uniform sampling requires n > 0")
    draw_index > 0 || error("draw_index must be positive")
    modulus = UInt64(n)
    threshold = mod(-modulus, modulus)
    attempt = 0
    while true
        digest = sha256(length_prefixed_payload(version, seed, draw_index, attempt))
        value = zero(UInt64)
        for byte in digest[1:8]
            value = (value << 8) | UInt64(byte)
        end
        value >= threshold && return Int(mod(value, modulus)) + 1
        attempt += 1
    end
end

"""Canonical text representation used to hash an effective configuration."""
function canonical_value(value)
    if value isa AbstractDict
        keys_sorted = sort!(string.(collect(keys(value))))
        parts = String[]
        for key in keys_sorted
            original_key = first(k for k in keys(value) if string(k) == key)
            push!(parts, repr(key) * ":" * canonical_value(value[original_key]))
        end
        return "{" * join(parts, ",") * "}"
    elseif value isa AbstractVector
        return "[" * join(canonical_value.(value), ",") * "]"
    elseif value isa AbstractString
        return repr(String(value))
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value isa Integer
        return string(value)
    elseif value isa AbstractFloat
        return isfinite(value) ? repr(Float64(value)) : error("Non-finite configuration value")
    elseif value === nothing
        return "null"
    end
    return repr(value)
end

configuration_sha256(raw::AbstractDict) = bytes2hex(sha256(codeunits(canonical_value(raw))))

"""Return the current Git commit and complete worktree dirty state."""
function git_provenance(repo_root::AbstractString)
    isdir(repo_root) || error("Repository root does not exist: $repo_root")
    commit = try
        readchomp(`git -C $repo_root rev-parse HEAD`)
    catch err
        error("Cannot determine Git commit for $repo_root: $(sprint(showerror, err))")
    end
    status = readchomp(`git -C $repo_root status --porcelain --untracked-files=normal`)
    return (commit = commit, dirty = !isempty(strip(status)))
end

"""Normalize path separators for portable CSV provenance fields."""
portable_path(path::AbstractString) = replace(normpath(path), '\\' => '/')

"""Compare paths with Windows case folding while preserving POSIX semantics."""
function same_path(a::AbstractString, b::AbstractString)
    left = normpath(abspath(a))
    right = normpath(abspath(b))
    return Sys.iswindows() ? lowercase(left) == lowercase(right) : left == right
end

"""Read a CSV file into string dictionaries."""
function read_csv_dicts(path::AbstractString)
    isfile(path) || error("Missing CSV file: $path")
    lines = readlines(path)
    isempty(lines) && error("CSV file is empty: $path")
    header = parse_csv_line(lines[1])
    rows = Dict{String, String}[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        fields = parse_csv_line(line)
        length(fields) == length(header) || error("Malformed CSV row in $path")
        push!(rows, Dict(name => value for (name, value) in zip(header, fields)))
    end
    return rows
end

"""Parse one RFC-4180-style CSV record without adding an external dependency."""
function parse_csv_line(line::AbstractString)
    fields = String[]
    buffer = IOBuffer()
    in_quotes = false
    index = firstindex(line)
    while index <= lastindex(line)
        char = line[index]
        if char == '"'
            next_index = nextind(line, index)
            if in_quotes && next_index <= lastindex(line) && line[next_index] == '"'
                write(buffer, '"')
                index = nextind(line, next_index)
                continue
            end
            in_quotes = !in_quotes
        elseif char == ',' && !in_quotes
            push!(fields, String(take!(buffer)))
        else
            write(buffer, char)
        end
        index = nextind(line, index)
    end
    in_quotes && error("Unterminated quoted CSV field")
    push!(fields, String(take!(buffer)))
    return fields
end

function csv_escape(value::AbstractString)
    escaped = replace(value, "\"" => "\"\"")
    if occursin(',', escaped) || occursin('"', escaped) || occursin('\n', escaped)
        return "\"" * escaped * "\""
    end
    return escaped
end

"""Write a deterministic CSV file."""
function write_csv(path::AbstractString, header::Vector{String}, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(csv_escape.(header), ','))
        for row in rows
            length(row) == length(header) || error("CSV row length mismatch for $path")
            println(io, join(csv_escape.(string.(row)), ','))
        end
    end
    return path
end

function required_csv_field(row::Dict{String, String}, key::AbstractString)
    haskey(row, key) || error("CSV row is missing required field '$key'")
    value = strip(row[key])
    isempty(value) && error("CSV field '$key' may not be empty")
    return value
end

parse_bool(value::AbstractString) = lowercase(strip(value)) in ("true", "1", "yes", "y")

"""Sort IDs such as `s03_c012` by scenario and within-scenario case."""
function geology_sort_key(geology_id::AbstractString)
    match_result = match(r"^s(\d+)_c(\d+)$", geology_id)
    match_result === nothing && return (typemax(Int), typemax(Int), String(geology_id))
    return (parse(Int, match_result.captures[1]),
            parse(Int, match_result.captures[2]),
            String(geology_id))
end
