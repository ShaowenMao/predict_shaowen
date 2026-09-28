# Reservoir comparison result locations

Verified on Engaging, 2026-09-28. The following base is physical **shared disk**,
not the shared flash namespace `/orcd/compute/juanes/001`:

```text
/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908
```

## Corrected method and angle comparisons

Under that base:

```text
corrected_hysteresis_15case_20260911_v1/variants/<variant>/results/
```

Fifteen variants: low/intermediate/high-risk cases, each with production control
at WRR30, MIP at WRR30 and WRR60, and hybrid at WRR30 and WRR60. The geological
cases are `s01_c018_case07`, `s03_c022_case12`, and `s03_c003_case08`.
Variant names include `low_mip_wrr60_fault_nohys` and
`intermediate_hybrid_wrr30_fault_nohys`. The `fault_nohys` label refers to
corrected reservoir hysteresis handling; it does not mean no pre-breakthrough Pc.
Host hysteresis is retained, PREDICT fault hysteresis disabled.

All 15 variant directories, their result roots and 30 recorded preflight/full
result directories exist and resolve under physical `/orcd/data/...`.
Exact identities, source paths and input hashes are in
`snapshot/corrected_reservoir/matrix.json`.

## Alternate pre-breakthrough treatment

```text
mip_pc_breakthrough_6case_20260916_v1/variants/<variant>_pczero/results/
```

Six matched MIP comparisons: the same three geological cases at 30 and 60
degrees. These apply the Pc-only zero-before-breakthrough convention. All six
variant directories, result roots and 12 preflight/full result directories
exist and resolve under physical `/orcd/data/...`.
See `snapshot/pc_onset/matrix.json` and `sources.json` for exact pairings and
input checksums. Together these two corrected studies comprise **21 variants**.

## Earlier studies and inputs

Earlier comparison references and paired physical/zero Pc inputs are under:

```text
comparative_3case_wrr30_60_pc_conventions_20260910_v1/
comparative_3case_wrr30_60_pc_conventions_20260910_v1/reservoir_matrix_launch_20260911_v1/
hybrid_low_high_wrr30_60_20260911_v1/
```

The corrected matrices also retain explicit references to the earlier
intermediate-case MIP/hybrid studies. Use those matrices rather than inventing
a replacement path. This inspection verifies storage locations, not a fresh
checksum audit or completion status of every result inside every old study.

The separate large production reservoir campaign is archived under:

```text
/orcd/data/juanes/001/shaowen/gom_full_production/campaigns/
```

Its working output uses `/orcd/scratch/orcd/012/shaowen/gom_grid`, distinct from
both project disk and shared flash. None of these locations is a cleanup target
of this source publication. Preserve input datasets, runtime dependencies and
all comparison results.
