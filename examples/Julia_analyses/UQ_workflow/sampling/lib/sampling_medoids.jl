"""
    exact_logk_medoid(logK, candidate_rows, realization_ids)

Return the one-based source-library row minimizing the total Euclidean distance
to all candidate rows in physical three-dimensional log10-permeability space.
If exact cost ties occur, the smallest original realization ID is selected,
followed by the smallest source row as a final deterministic tie-break.
"""
function exact_logk_medoid(logK::AbstractMatrix{<:Real},
                           candidate_rows::AbstractVector{<:Integer},
                           realization_ids::AbstractVector{<:Integer})
    size(logK, 2) == 3 || error("Medoid calculation requires exactly three logK columns")
    size(logK, 1) == length(realization_ids) ||
        error("realization_ids length does not match logK")
    rows = sort!(unique(Int.(candidate_rows)))
    isempty(rows) && error("Cannot compute a medoid for an empty candidate set")
    all(1 .<= rows .<= size(logK, 1)) || error("Medoid candidate row is out of range")

    costs = zeros(Float64, length(rows))
    @inbounds for local_i in 1:(length(rows) - 1)
        row_i = rows[local_i]
        x1 = Float64(logK[row_i, 1])
        x2 = Float64(logK[row_i, 2])
        x3 = Float64(logK[row_i, 3])
        for local_j in (local_i + 1):length(rows)
            row_j = rows[local_j]
            d1 = x1 - Float64(logK[row_j, 1])
            d2 = x2 - Float64(logK[row_j, 2])
            d3 = x3 - Float64(logK[row_j, 3])
            distance = sqrt(d1 * d1 + d2 * d2 + d3 * d3)
            costs[local_i] += distance
            costs[local_j] += distance
        end
    end

    minimum_cost = minimum(costs)
    tied_local = findall(==(minimum_cost), costs)
    tied_rows = rows[tied_local]
    return first(sort(tied_rows; by = row -> (Int(realization_ids[row]), row)))
end

exact_logk_medoid(logK::AbstractMatrix{<:Real}, realization_ids::AbstractVector{<:Integer}) =
    exact_logk_medoid(logK, collect(1:size(logK, 1)), realization_ids)
