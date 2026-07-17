"""
    TexasFieldSlices

Slice draw-group utilities for the Texas offshore field application.
"""
module TexasFieldSlices

export build_slice_draw_groups,
       validate_slice_draw_groups,
       slice_group_rows

"""
    build_slice_draw_groups(num_slices, shared_slice_groups)

Build ordered draw groups for `num_slices` field slices.

Each slice normally receives its own draw group. Optional shared groups can be
provided to force selected slices to use the same sampled PREDICT realization.
"""
function build_slice_draw_groups(num_slices::Integer, shared_slice_groups::Vector)
    num_slices > 0 || error("num_slices must be positive")
    shared_by_slice = Dict{Int, Int}()
    normalized_groups = Vector{Vector{Int}}()
    for group in shared_slice_groups
        normalized = sort(unique(Int.(group)))
        length(normalized) >= 2 || error("Shared slice groups must contain at least two slices: $group")
        for slice in normalized
            1 <= slice <= num_slices || error("Shared slice $slice is outside 1:$num_slices")
            haskey(shared_by_slice, slice) && error("Slice $slice appears in more than one shared group")
            shared_by_slice[slice] = length(normalized_groups) + 1
        end
        push!(normalized_groups, normalized)
    end

    draw_groups = Vector{Vector{Int}}()
    consumed = Set{Int}()
    for slice in 1:num_slices
        slice in consumed && continue
        if haskey(shared_by_slice, slice)
            group = normalized_groups[shared_by_slice[slice]]
            push!(draw_groups, group)
            union!(consumed, group)
        else
            push!(draw_groups, [slice])
            push!(consumed, slice)
        end
    end

    validate_slice_draw_groups(num_slices, draw_groups)
    return draw_groups
end

"""
    validate_slice_draw_groups(num_slices, draw_groups)

Check that every slice appears exactly once.
"""
function validate_slice_draw_groups(num_slices::Integer, draw_groups::Vector{Vector{Int}})
    all_slices = sort(vcat(draw_groups...))
    expected = collect(1:num_slices)
    all_slices == expected || error("Slice draw groups do not cover exactly 1:$num_slices")
    return true
end

"""
    slice_group_rows(draw_groups)

Return CSV-ready rows for the slice-to-draw-group map.
"""
function slice_group_rows(draw_groups::Vector{Vector{Int}})
    rows = Vector{Vector{String}}()
    for (draw_group_index, slices) in enumerate(draw_groups)
        shared = length(slices) > 1 ? "true" : "false"
        for slice in slices
            push!(rows, [
                string(slice),
                string(draw_group_index),
                shared,
                join(slices, ";"),
            ])
        end
    end
    return sort(rows; by = row -> parse(Int, row[1]))
end

end
