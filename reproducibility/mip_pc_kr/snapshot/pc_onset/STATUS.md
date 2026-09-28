# Submission status

Six same-draw MIP Pc-onset comparisons submitted on 2026-09-16. All six native
preflights completed successfully in 88-118 seconds, with measured batch peak
memory of approximately 9.6-10.5 GiB. All six full simulations are now RUNNING
after their individual afterok dependencies cleared. No full simulation
completion is claimed.

| Risk | Angle | Preflight | Full simulation |
|---|---:|---|---|
| Low | 30 | 22847062_262 | 22847063_262 |
| Low | 60 | 22847064_262 | 22847065_262 |
| Intermediate | 30 | 22847066_1137 | 22847067_1137 |
| Intermediate | 60 | 22847068_1137 | 22847069_1137 |
| High | 30 | 22847070_848 | 22847071_848 |
| High | 60 | 22847072_848 | 22847073_848 |

All 3132 segment tables passed the isolated Pc-only audit: unchanged MIP gas/brine
Kr and saturation coordinates, unchanged post-breakthrough Pc nodes, and unchanged
other scientific datasets. All six physical donors are byte-identical to the
specific inputs of the matching completed corrected-hysteresis MIP runs.

The copied launcher configurations retain the same model, corrected runtime,
solver, tolerances and timestep controls. Julia parsing and negative validator
tests passed. Full assembled-model input and hysteresis checks passed in preflight.
A reporting-only final-check repair recognizes all 210 retained report states;
it does not relax scientific checks or alter simulation results.

Verified actual scheduling: mit_normal partition; mit_amf_advanced_cpu account
and QoS; 8 CPUs per job; 12 GiB preflight and 18 GiB full simulation. Full runs
request four days. Independent cases have no array concurrency limit.

Remote study root:

`/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/mip_pc_breakthrough_6case_20260916_v1`

All outputs, retained states, QoIs and temporary files use project disk, not
scratch. The filesystem reported 36 TiB available during preparation.
Original production and all completed comparison code, inputs, results and jobs
remain untouched. Existing automatic-check settings were not changed.
