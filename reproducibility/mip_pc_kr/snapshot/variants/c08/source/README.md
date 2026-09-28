# Original 12-case MIP reservoir matrix

Launch only the ten missing variants; retain existing intermediate WRR30 jobs
22438751_1137 and 22474505_1137. Preserve production, hybrid studies, and all
accepted fine-core outputs. New files live under the comparative study's
reservoir_matrix_launch_20260911_v1 directory on project storage.

Reuse the five remaining case/angle datasets, each with 522 accepted 100-state
MIP/dip-parallel-Kr curves. Verify receipt hashes and original draw mappings.
The physical Pc export retains native positive-saturation nodes and exact inlet
entry pressure. The zero-before-breakthrough export follows the already-running
intermediate WRR30 intervention: exact minimax breakthrough at zero saturation,
linear onset to the first sampled post-breakthrough point, with the same Sg/Kr
knots and post-breakthrough branch. This is not a new simulator entry switch.

The two Pc conventions have identical Kr values and interpolation grids. No
MIP sweeps, phase-flow upscaling, smoothing, new saturation endpoints, or
production dynamic-Kr approximations are introduced. The existing export
converter permits only its documented 1e-9 absolute solver-roundoff correction.
Unresolved structural errors remain blocking, never silently accepted.

Five independent export jobs feed ten independent native qualification/preflight
jobs and ten full reservoir jobs. Dependencies use afterok, without throttling
arrays or changing production scheduling. Full runs retain original 8-CPU/18-GB
settings, fresh t=0, matched solver/physics, QoI4, all 210 report states, and two
rolling checkpoints on project storage. Preflights use 8 CPUs/24 GB. Small export
jobs use 2 CPUs/18 GB; exact threshold extraction and all simulations run on compute
nodes. Source, production input, accepted receipt, and exported table hashes are
recorded. A full run cannot start if its native checks fail.
