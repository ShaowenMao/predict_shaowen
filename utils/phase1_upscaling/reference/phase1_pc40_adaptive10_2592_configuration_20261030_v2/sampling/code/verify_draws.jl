using SHA

# Read the original, unchanged Julia implementation as an independent oracle.
include("D:/Github/predict_shaowen_independent_full_fault_v1/examples/Julia_analyses/UQ_workflow/sampling/lib/sampling_provenance.jl")
root = normpath(joinpath(@__DIR__, ".."))
version = "sha256_scenario_pc_u63_counter_u64_rejection_v1"
rows = read_csv_dicts(joinpath(root, "phase1", "sampling_streams.csv"))
@assert length(rows) == 15552
count = 0
for row in rows
    seed = stable_case_seed(20261029, row["scenario_id"], "phase1", "independent_full",
                            parse(Int, row["replicate_id"]), row["window_id"]; version=version)
    @assert seed == parse(UInt64, row["sampling_seed"])
    picks = [stable_uniform_index(seed, j, parse(Int, row["n_accepted"]); version=version) for j in 1:87]
    @assert bytes2hex(sha256(join(picks, ','))) == row["source_rows_sha256"]
    global count += 87
end
println("PASS: original Julia implementation reproduced all ", length(rows), " seeds and ", count, " source-row draws.")
