"""
    Level2Plotting

Shared plotting and table-reading helpers for Level 2 review scripts.

The module centralizes component labels, state colors, cluster colors, MAT
value conversion helpers, PCA projection, cluster-order utilities, and simple
CSV readers used by the figure scripts.
"""
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
       ordered_cluster_joint_rank_medians,
       ordered_cluster_score_medians,
       joint_rank_score_range,
       state_score_range,
       state_component_summary,
       medoid_component_values,
       perturbation_pool_distance_profile,
       read_simple_csv,
       csv_numeric_column,
       csv_string_column,
       state_mean_matrix,
       metric_label

const COMPONENT_NAMES = ("kxx", "kyy", "kzz")
const COMPONENT_LABELS = ("log10(kxx)", "log10(kyy)", "log10(kzz)")
const STATE_ORDER = ("low", "high")
const STATE_COLORS = Dict(
    "low" => RGBf(0.153, 0.392, 0.780),
    "high" => RGBf(0.820, 0.235, 0.196),
)
const CLUSTER_COLORS = [
    RGBf(0.000, 0.255, 0.650),  # deep blue
    RGBf(0.930, 0.460, 0.000),  # orange
    RGBf(0.000, 0.620, 0.300),  # green
    RGBf(0.090, 0.090, 0.090),  # charcoal
    RGBf(0.610, 0.450, 0.000),  # olive brown
]

"""
    activate_plot_theme!()

Activate CairoMakie and set the default Level 2 plotting theme.
"""
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

"""
    float_scalar(value)

Convert a scalar or array-like MAT-loaded value to `Float64`.
"""
float_scalar(value) = value isa AbstractArray ? Float64(first(vec(value))) : Float64(value)

"""
    int_scalar(value)

Convert a scalar or array-like MAT-loaded value to `Int`.
"""
int_scalar(value) = Int(round(float_scalar(value)))

"""
    vector_int(values)

Convert scalar or array-like MAT-loaded values to a vector of integers.
"""
vector_int(values) = values isa AbstractArray ? vec(Int.(values)) : [Int(values)]

"""
    vector_float(values)

Convert scalar or array-like MAT-loaded values to a vector of floats.
"""
vector_float(values) = values isa AbstractArray ? vec(Float64.(values)) : [Float64(values)]

"""
    matrix_float(values)

Convert array-like MAT-loaded values to a `Matrix{Float64}`.
"""
matrix_float(values) = Matrix{Float64}(values)

"""
    sample_indices(n, max_points, seed)

Return deterministic sample indices for plotting at most `max_points` rows.
"""
function sample_indices(n::Int, max_points::Int, seed::Int)
    n <= max_points && return collect(1:n)
    rng = MersenneTwister(seed)
    return sort(randperm(rng, n)[1:max_points])
end

"""
    pca_projection(z)

Project rows of `z` onto the first two principal-component axes and return
both coordinates and explained-variance fractions.
"""
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

"""
    cluster_rank_assignments(state)

Map original cluster ids to ordered cluster numbers for every sample.
"""
function cluster_rank_assignments(state::Dict{String, Any})
    assignments = vector_int(state["cluster_assignments"])
    order = vector_int(state["cluster_order"])
    rank_map = Dict(cluster_id => rank for (rank, cluster_id) in enumerate(order))
    return [rank_map[cluster_id] for cluster_id in assignments]
end

"""
    ordered_cluster_sizes(state)

Return cluster sizes in low-to-high ordered-cluster order.
"""
function ordered_cluster_sizes(state::Dict{String, Any})
    sizes = vector_int(state["cluster_sizes"])
    order = vector_int(state["cluster_order"])
    return [sizes[cluster_id] for cluster_id in order]
end

"""
    ordered_cluster_joint_rank_medians(state)

Return cluster median joint rank scores in ordered-cluster order.
"""
function ordered_cluster_joint_rank_medians(state::Dict{String, Any})
    medians = haskey(state, "cluster_joint_rank_medians") ?
        vector_float(state["cluster_joint_rank_medians"]) :
        vector_float(state["cluster_score_medians"])
    order = vector_int(state["cluster_order"])
    return [medians[cluster_id] for cluster_id in order]
end

"""
    ordered_cluster_score_medians(state)

Backward-compatible alias for `ordered_cluster_joint_rank_medians`.
"""
ordered_cluster_score_medians(state::Dict{String, Any}) =
    ordered_cluster_joint_rank_medians(state)

"""
    joint_rank_score_range(state, label)

Return minimum, median, and maximum joint rank score within a low/high state.
"""
function joint_rank_score_range(state::Dict{String, Any}, label::AbstractString)
    scores = haskey(state, "joint_rank_score") ?
        vector_float(state["joint_rank_score"]) :
        vector_float(state["state_score"])
    indices = vector_int(state["$(label)_indices"])
    values = scores[indices]
    return (
        minimum(values),
        median(values),
        maximum(values),
    )
end

"""
    state_score_range(state, label)

Backward-compatible alias for `joint_rank_score_range`.
"""
state_score_range(state::Dict{String, Any}, label::AbstractString) =
    joint_rank_score_range(state, label)

"""
    state_component_summary(state, label)

Return component-wise medians and interquartile ranges for a low/high state.
"""
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

"""
    medoid_component_values(state, label)

Return the `log10(k)` component vector for a state's medoid realization.
"""
function medoid_component_values(state::Dict{String, Any}, label::AbstractString)
    log_perms = matrix_float(state["log_perms"])
    medoid_index = int_scalar(state["$(label)_medoid_index"])
    return vec(log_perms[medoid_index, :])
end

"""
    perturbation_pool_distance_profile(state, label, pool)

Return sorted distances from a state medoid to all samples in a perturbation
pool.
"""
function perturbation_pool_distance_profile(state::Dict{String, Any}, label::AbstractString, pool::AbstractString)
    features = state_distance_features(state)
    medoid_index = int_scalar(state["$(label)_medoid_index"])
    indices = vector_int(state["$(label)_$(pool)_pool"])
    medoid = vec(features[medoid_index, :])
    distances = [norm(vec(features[idx, :]) - medoid) for idx in indices]
    return sort(distances)
end

"""
    state_distance_features(state)

Reconstruct the feature matrix used for Level 2 distance-based diagnostics.
"""
function state_distance_features(state::Dict{String, Any})
    metric = String(get(state, "distance_metric", "local_normal"))
    log_perms = matrix_float(state["log_perms"])
    local_normal_scores = matrix_float(state["local_normal_scores"])
    scales = haskey(state, "distance_component_scales") ?
        vector_float(state["distance_component_scales"]) :
        ones(Float64, size(log_perms, 2))
    weights = haskey(state, "distance_weights") ?
        vector_float(state["distance_weights"]) :
        ones(Float64, size(log_perms, 2))

    values = metric == "local_normal" ? local_normal_scores : log_perms
    features = similar(values)
    for j in axes(values, 2)
        scale = scales[j] > 0 ? scales[j] : 1.0
        features[:, j] .= values[:, j] .* sqrt(weights[j]) ./ scale
    end
    return features
end

"""
    state_mean_matrix(state)

Return a two-row matrix of low/high mean `log10(k)` vectors.
"""
function state_mean_matrix(state::Dict{String, Any})
    matrix = zeros(Float64, length(STATE_ORDER), 3)
    for (i, label) in enumerate(STATE_ORDER)
        matrix[i, :] .= vector_float(state["$(label)_mean_log_perm"])
    end
    return matrix
end

"""
    metric_label(name)

Return a readable label for validation and diagnostic metric names.
"""
function metric_label(name::AbstractString)
    labels = Dict(
        "same_k_rate" => "same K",
        "same_unimodality_rate" => "same unimodality",
        "mean_abs_silhouette_delta" => "mean |silhouette delta|",
        "mean_global_medoid_distance" => "mean global medoid dist.",
        "mean_low_medoid_distance" => "mean low medoid dist.",
        "mean_high_medoid_distance" => "mean high medoid dist.",
        "abs_silhouette_delta" => "|silhouette delta|",
        "global_medoid_distance" => "global medoid dist.",
        "low_medoid_distance" => "low medoid dist.",
        "high_medoid_distance" => "high medoid dist.",
    )
    return get(labels, String(name), String(name))
end

"""
    read_simple_csv(path)

Read a simple CSV file into `(header, rows)` without external dependencies.
"""
function read_simple_csv(path::AbstractString)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    isempty(lines) && error("CSV file is empty: $path")
    header = parse_csv_line(lines[1])
    rows = [parse_csv_line(line) for line in lines[2:end]]
    return (header = header, rows = rows)
end

"""
    csv_numeric_column(table, name)

Extract and parse one numeric column from a table returned by `read_simple_csv`.
"""
function csv_numeric_column(table, name::AbstractString)
    idx = findfirst(==(String(name)), table.header)
    idx === nothing && error("Column not found in CSV: $name")
    return [parse(Float64, row[idx]) for row in table.rows]
end

"""
    csv_string_column(table, name)

Extract one string column from a table returned by `read_simple_csv`.
"""
function csv_string_column(table, name::AbstractString)
    idx = findfirst(==(String(name)), table.header)
    idx === nothing && error("Column not found in CSV: $name")
    return [String(row[idx]) for row in table.rows]
end

"""
    parse_csv_line(line)

Split one CSV line while respecting quoted fields.
"""
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
