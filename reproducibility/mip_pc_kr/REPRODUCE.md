# Reproduction guide

Run commands from `reproducibility/mip_pc_kr`. Use new private output directories.
The commands below do not submit Slurm jobs, create claims, import into a live
ledger or publish acceptance to a campaign. Never run the numerical examples
on an Engaging login node. The supplied small fixtures may be tested locally.

## Default selection

Unless an alternative is explicitly requested, use **WRR60 Pc100/Kr10 with
physical pre-breakthrough behavior**: 60-degree clay Pc/Kr, 100 saved Pc
pressure points and at most 10 selected Kr states from the same MIP microstates.
Follow sections 3 and 4 with verified `planned60` inputs. The `flow` and `export`
commands are the default numerical/export path; section 5's `pc-onset` command
is opt-in and must not be appended automatically. Historical WRR30/full100
paths are preserved alternatives, not defaults. Existing acceptance, resource
and preservation safeguards still apply without change.

## 1. Environments and source verification

```sh
python workflow.py verify
python3.10 -m venv /new/private/path/mip-python
/new/private/path/mip-python/bin/pip install -r snapshot/python_environment/pip_freeze.txt/pip_freeze.txt
```

Qualified exporter versions: Python 3.10, NumPy 2.2.6, SciPy 1.15.3 and h5py
3.14.0. The original environment installation report is retained. New adapters
support Python 3.10+; the pure Python tests were run with the pinned libraries.

Numerics require **Julia 1.12.6**, IEEE math, bounds checks, two Julia/BLAS
threads, one GC thread, and the included empty project plus vendored implicit
dependencies. The managed backend checks the exact UMFPACK wrapper source hash.
The Linux campaign bridge is `snapshot/bridge_reference/libphase_bridge.so`;
its original SHA256 is in `SOURCE_MANIFEST.json`. Its build record and C++ sources
are included. On another platform, rebuilding is possible but does not establish
bitwise identity or production qualification:

```sh
python snapshot/pipeline/vendor/kr/build_bridge.py snapshot/pipeline/vendor/numeric /new/private/path/bridge
```

The unchanged builder requires GNU C++ and quadmath, disables fast-math and
floating-point contraction, and records compiler/source/library hashes.
The `flow` adapter intentionally requires the original pinned Linux bridge.
It does not silently substitute an unqualified rebuilt binary.

Native reservoir import uses Julia **1.10.4**, the included corrected
`snapshot/reservoir_runtime` source with its `Project.toml`/`Manifest.toml`, and
the corresponding Jutul dependencies. Instantiate a private environment/depot
from that manifest; do not update dependencies or overwrite the installed runtime
used by running jobs. Historical launch configurations also record absolute
cluster paths; relocate those operational bindings to the archived sources,
not an arbitrary newer checkout. Corrected runtime identity:
`c403d9dd1bbb424ddfae9ae659d900c26c84e59d` in the comparison records. The exact
archived source bytes, not a possibly different working tree, are authoritative.

## 2. Original inputs and exact replay

Keep these physical-disk trees protected:

```text
/orcd/data/juanes/001/shaowen/predict_shaowen/production_freezes/collapsed_cell_union_20260722_v7/inputs/predict
/orcd/data/juanes/001/shaowen/predict_shaowen/handoffs/independent_full_fault_v1_phase1_20260809_v1
/orcd/data/juanes/001/shaowen/predict_shaowen/production_runs/independent_full_fault_v1_phase1_20260809_v1
/orcd/data/juanes/001/shaowen/gom_full_production/inputs
```

Input inventories and phase-1 metadata are preserved in `snapshot/original_inputs`
and `snapshot/phase1_inputs`. The phase-1 scope is 2,430 cases, 162 geologies,
522 positions per case; the older July freeze alone describes 1,620 cases.
Do not confuse these scopes. No input payloads were removed or relocated.

Replay requires the exact checkpoint, selected-sample index, replay seed,
original sampler draw identity, geometry summary and case/slot mapping. The
MATLAB geometry entry point is:

```matlab
addpath('/checkout/reproducibility/mip_pc_kr/snapshot/replay/frozen');
prepare_slot(selection_csv, summary_json, fresh_output, physics_root, checkpoint_root, mrst_root);
```

It requires a compute allocation and MATLAB R2025b Update 5 for the qualified
environment, MRST 2025a checkout `75749fa`, and PREDICT physics commit
`68351e35f3679317b35532a9ca0533674e0aafb5`. That PREDICT commit is available in
the repository history; do not use unrelated uncommitted edits. Full loaded-source
checksums are in `snapshot/replay/runtime_inventory.json`. MATLAB/MRST and the
checkpoint payloads are external dependencies, not redistributed runtime images.
Keep the recorded MATLAB dependencies/modules and license available.

The existing assignment/replay verification helpers and source locks are archived
in `snapshot/replay`. Successful output must preserve architecture, pore volume,
intrinsic properties, exact draws and the original tolerance checks. A seed alone
is not sufficient reproduction evidence.

## 3. Pressure planning and WRR60 preparation

Provide a JSON input specification with `raw` and `geometry` entries. Each is an
object containing `path` and the expected lowercase `sha256` from the retained
receipt. Do not generate a replacement checksum to accept an unexpectedly changed
source. Use the original unplanned replay binaries, not a different realization.

```sh
python workflow.py prepare /inputs/unplanned.json /new/private/path/prepared --julia /path/to/julia-1.12.6
```

The adapter calls the unchanged fixed100 planner and WRR60 transform, producing
`planned30` and `planned60`. Original geometry and physical fields remain
unchanged; the same endpoint/pressure coverage safeguards run. The older WRR30
and full100 numerical paths remain in the frozen historical sources. Do not use
the WRR60-only sparse worker for a WRR30 input.

## 4. Default WRR60 Pc100/Kr10 flow and physical export

The flow JSON needs `raw`, `geometry`, `scenario_audit`, and `bridge`, each with
`path` and expected `sha256`, plus `slot` and a qualified `memory_class`.
Use the verified `planned60` outputs and the included reference Linux bridge.

Supported class memory in MiB is unchanged: `mixed_60000:2816`,
`mixed_90000:7040`, `mixed_120000:13952`, `mixed_150000:20096`,
`clay_120000:3712`. Two CPUs per numerical worker. The class must be established
from the existing geometry/contrast qualification, not chosen to fit a desired
allocation. Unknown classes remain held; this adapter does not qualify them.

```sh
python workflow.py flow /inputs/planned60.json /new/private/path/flow --julia /path/to/julia-1.12.6
python workflow.py export /new/private/path/flow/curves/core /new/private/path/export
```

The flow adapter primes only its private cache, then invokes the original worker
with strict existing-cache mode. It validates all frozen sources, binary inputs,
microstate equality, corrected-flow receipts and native factor lifetimes.
An exit code alone is not science acceptance: `CORE_PASS.toml`, export receipts,
native constitutive checks and full-case mapping validation are still required.
The exporter retains physical pre-breakthrough Pc and the prescribed bounded
noise correction. It refuses changed artifacts or unknown interior plateaus.

Native constitutive import can be performed using the preserved importer:

```sh
julia-1.10.4 --startup-file=no --threads=1 --project=/private/corrected-runtime snapshot/full_fault_export/native.jl /new/private/path/native /new/private/path/export/sgof.csv
```

For a full case, use the preserved 522-position assembly/native verification
sources and original mapping manifests. Four or fewer core tests are not a
substitute for full-case mapping verification.

## 5. Opt-in alternate Pc treatment without recomputing Kr

The exact historical alternative uses three input tables: the accepted `mip`
specific MAT file, its matching physical-Pc donor, and its matching zero-before-
breakthrough donor. `snapshot/pc_onset/sources.json` records the six real pairs,
original paths, bytes and checksums. The historical exporter that produced the
paired donors is retained under `snapshot/historical_releases`.

In `historical_releases/mip_matrix_launch_20260911_v1`, the complete archived
`study.py:export_dataset` procedure validates the 522 original accepted receipts,
builds the physical tables, runs `extract_breakthrough.jl` on the checksum-pinned
raw manifest and entry-pressure metadata, and calls `vendor/build_pc_only.py`
to construct the alternative. `make_specific.m` preserves the original case
mapping when writing both donor MAT files. This is the recipe for reproducing
the donors; its historical absolute paths must be relocated in a private copy.
Do not run its old scheduling or submission entry points. Breakthrough recovery
requires a compute allocation and rejects a threshold beyond the qualified
pre-mobility-loss regime rather than inventing an approximation. Do not silently
substitute Pc100/Kr10 interpolated curves for a recorded full100 donor.

Provide a JSON specification with `mip`, `physical`, and `zero` objects, each
with `path` and expected `sha256`:

```sh
python workflow.py pc-onset /inputs/pc-onset.json /new/private/path/pc-onset
```

This calls the original Pc-only converter in a private directory. All 522 real
region mappings, sample identities and seeds must match. Both Kr columns,
saturation knots, post-breakthrough Pc nodes and other scientific datasets are
checked for exact preservation. Existing input files are only read/copied, never
edited. Output is `inputs/case/candidate_specific.mat` with
`pc_only_audit.json`. Missing donors are a blocker, not permission to approximate
breakthrough pressure or rerun accepted flow calculations. Original pair-input
and corrected-hysteresis tests are preserved with the reservoir comparison code.

## 6. Scheduling and preservation

For any separately authorized Engaging CPU submission, explicitly request
partition `mit_normal` and both account/QoS `mit_amf_advanced_cpu`. Inspect the
actual account, QoS, CPUs and memory while held, before release. Apply fresh
shared admission, pending/reservation reconciliation, storage guards and measured
memory policies. The shared ceiling is 500 CPUs and 2,031,616 MiB, not an
entitlement to reserve all of it. This source package has no submission command
and must not bypass the qualified production admission mechanism.

Original campaign claims, ledgers, source locks and completed work remain
untouched. The source publication and fixture tests do not authorize broad flash
deletion, certify every external dependency as backed up, or change any live
production manager.
