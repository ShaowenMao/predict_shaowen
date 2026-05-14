"""
    step01_load_window_library(manifest_row)

Load one window's PREDICT proxy library from a manifest row.

The manifest row is expected to come from `Level2IO.read_manifest_csv` and
contains the geology id, window name, sample kind, and resolved MAT-file path.
The returned dictionary contains both raw permeability values and their
`log10(k)` transform, along with source metadata used by later workflow steps.
"""
function step01_load_window_library(manifest_row::Dict{String, String})
    return Level2IO.load_proxy_library(manifest_row)
end
