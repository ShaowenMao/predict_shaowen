module Level2Plotting

using CairoMakie
using Statistics
using LinearAlgebra
using Random

export COMPONENT_NAMES,
       COMPONENT_LABELS,
       STATE_ORDER,
       STATE_COLORS,
       CLUSTER_COLORS,
       activate_plot_theme!,
       float_scalar,
       int_scalar,
       vector_int,
       vector_float,
       matrix_float,
       sample_indices,
       pca_projection,
       cluster_rank_assignments,
       ordered_cluster_sizes,
       ordered_cluster_score_medians,
       state_score_range,
       state_component_summary,
       medoid_component_values,
       neighbor_distance_profile,
       read_simple_csv,
       csv_numeric_column,
       csv_string_column,
       state_mean_matrix,
       metric_label

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx)", "log10(kyy)", "log10(kzz)")
const STATE_ORDER = ("low", "central", "high")
const STATE_COLORS = Dict(
    "low" => RGBf(0.153, 0.392, 0.780),
    "central" => RGBf(0.420, 0.420, 0.420),
    "high" => RGBf(0.820, 0.235, 0.196),
)
const CLUSTER_COLORS = [
    RGBf(0.153, 0.392, 0.780),
    RGBf(0.820, 0.235, 0.196),
    RGBf(0.212, 0.620, 0.365),
    RGBf(0.702, 0.518, 0.102),
    RGBf(0.500, 0.353, 0.831),
]

function activate_plot_theme!()
    CairoMakie.activate!()
    set_theme!(Theme(
        fontsize = 16,
        figure_padding = 16,
        Axis = (
            titlesize = 18,
            xlabelsize = 16,
            ylabelsize = 16,
            xticklabelsize = 12,
            yticklabelsize = 12,
            titlegap = 8,
            xgridvisible = false,
            ygridvisible = true,
            backgroundcolor = RGBf(0.985, 0.985, 0.985),
        ),
        Legend = (
            framevisible = false,
            labelsize = 13,
            titlesize = 14,
        ),
    ))
end

float_scalar(value) = value isa AbstractArray ? Float64(first(vec(value))) : Float64(value)
int_scalar(value) = Int(round(float_scalar(value)))
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]
vector_float(values) = values isa AbstractArray ? vec(Float64.(values)) : [Float64(values)]
matrix_float(values) = Matrix{Float64}(values)

function sample_indices(n::Int, max_points::Int, seed::Int)
    n <= max_points && return collect(1:n)
    rng = MersenneTwister(seed)
    return sort(randperm(rng, n)[1:max_points])
end

function pca_projection(z::Matrix{Float64})
    centered = z .- mean(z; dims = 1)
    _, s, v = svd(centered)
    n_axes = min(2, size(v, 2))
    coords = centered * v[:, 1:n_axes]
    if n_axes == 1
        coords = hcat(coords[:, 1], zeros(size(coords, 1)))
    end
    total = sum(abs2, s)
    explained = total > 0 ? (abs2.(s[1:n_axes]) ./ total) : zeros(Float64, n_axes)
    if length(explained) == 1
        explained = [explained[1], 0.0]
    end
    return coords, explained
end

function cluster_rank_assignments(state::Dict{String, Any})
    assignments = vector_int(state["cluster_assignments"])
    order = vector_int(state["cluster_order"])
    rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(order))
    return [rank_map[cluster_id] for cluster_id in assignments]
end

function ordered_cluster_sizes(state::Dict{String, Any})
    sizes = vector_int(state["cluster_sizes"])
    order = vector_int(state["cluster_order"])
    return [sizes[cluster_id] for cluster_id in order]
end

function ordered_cluster_score_medians(state::Dict{String, Any})
    medians = vector_float(state["cluster_score_medians"])
    order = vector_int(state["cluster_order"])
    return [medians[cluster_id] for cluster_id in order]
end

function state_score_range(state::Dict{String, Any}, label::AbstractString)
    scores = vector_float(state["state_score"])
    indices = vector_int(state["$(label)_indices"])
    values = scores[indices]
    return (
        minimum(values),
        median(values),
        maximum(values),
    )
end

function state_component_summary(state::Dict{String, Any}, label::AbstractString)
    log_perms = matrix_float(state["log_perms"])
    indices = vector_int(state["$(label)_indices"])
    values = log_perms[indices, :]
    medians = [median(view(values, :, j)) for j in 1:size(values, 2)]
    q25 = [quantile(view(values, :, j), 0.25) for j in 1:size(values, 2)]
    q75 = [quantile(view(values, :, j), 0.75) for j in 1:size(values, 2)]
    return (
        medians = medians,
        q25 = q25,
        q75 = q75,
    )
end

function medoid_component_values(state::Dict{String, Any}, label::AbstractString)
    log_perms = matrix_float(state["log_perms"])
    medoid_index = int_scalar(state["$(label)_medoid_index"])
    return vec(log_perms[medoid_index, :])
end

function neighbor_distance_profile(state::Dict{String, Any}, label::AbstractString, neighborhood::AbstractString)
    z = matrix_float(state["local_normal_scores"])
    medoid_index = int_scalar(state["$(label)_medoid_index"])
    indices = vector_int(state["$(label)_$(neighborhood)_neighbors"])
    medoid = vec(z[medoid_index, :])
    distances = [norm(vec(z[idx, :]) - medoid) for idx in indices]
    return sort(distances)
end

function state_mean_matrix(state::Dict{String, Any})
    matrix = zeros(Float64, length(STATE_ORDER), 3)
    for (i, label) in enumerate(STATE_ORDER)
        matrix[i, :] .= vector_float(state["$(label)_mean_log_perm"])
    end
    return matrix
end

function metric_label(name::AbstractString)
    labels = Dict(
        "same_k_rate" => "same K",
        "same_unimodality_rate" => "same unimodality",
        "mean_abs_silhouette_delta" => "mean |silhouette delta|",
        "mean_global_medoid_distance" => "mean global medoid dist.",
        "mean_low_medoid_distance" => "mean low medoid dist.",
        "mean_high_medoid_distance" => "mean high medoid dist.",
        "mean_central_medoid_distance" => "mean central medoid dist.",
        "abs_silhouette_delta" => "|silhouette delta|",
        "global_medoid_distance" => "global medoid dist.",
        "low_medoid_distance" => "low medoid dist.",
        "high_medoid_distance" => "high medoid dist.",
        "central_medoid_distance" => "central medoid dist.",
    )
    return get(labels, String(name), String(name))
end

function read_simple_csv(path::AbstractString)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && error("CSV file is empty: $path")
    header = parse_csv_line(lines[1])
    rows = [parse_csv_line(line) for line in lines[2:end]]
    return (header = header, rows = rows)
end

function csv_numeric_column(table, name::AbstractString)
    idx = findfirst(==(String(name)), table.header)
    idx === nothing && error("Column not found in CSV: $name")
    return [parse(Float64, row[idx]) for row in table.rows]
end

function csv_string_column(table, name::AbstractString)
    idx = findfirst(==(String(name)), table.header)
    idx === nothing && error("Column not found in CSV: $name")
    return [String(row[idx]) for row in table.rows]
end

function parse_csv_line(line::AbstractString)
    values = String[]
    buffer = IOBuffer()
    in_quotes = false
    i = firstindex(line)
    while i <= lastindex(line)
        c = line[i]
        if c == '"'
            if in_quotes && i < lastindex(line) && line[nextind(line, i)] == '"'
                print(buffer, '"')
                i = nextind(line, i)
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(values, String(take!(buffer)))
        else
            print(buffer, c)
        end
        i = nextind(line, i)
    end
    push!(values, String(take!(buffer)))
    return values
end

end
