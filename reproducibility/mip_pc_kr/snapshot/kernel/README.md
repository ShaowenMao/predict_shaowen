# Isolated Pc100 / Kr10 Warm-Worker Qualification

Only this bounded pilot is authorized. No campaign refill or production changes.

Kernel.jl is extracted verbatim from the successful eight-core benchmark; only
the calling sequence changes. A worker executes sparse10 once per core, never
the benchmark's cold10/full100/warm10 sequence. All 100 saved MIP state hashes,
Pc rows and event diagnostics are compared with immutable saved controls.
Selected flow values, zeros, factor cleanup and exact entry are checked.

Small batch: intermediate100, high144, intermediate98, intermediate100 sentinel.
Larger batch: low125, low89, intermediate167, high165; the 90k-cell demanding
core is third. The small batch also has a controlled two-core stop followed by
a dependency-linked recovery that verifies/skips the completed prefix.

Memory requests: bootstrap 4608 MiB; small/prefix/resume 2816 MiB each; larger
6656 MiB; collector 768 MiB; native import 3072 MiB. Numerical jobs use two CPUs;
collector/native use one. These are targeted pilot reservations, not updated
campaign-wide classes. Whole-job cgroup peaks include Python, Julia and page
cache. Require no memory-limit events and the inherited 5% acceptance headroom.
Future reservations will use measured envelopes and 20% relative headroom.

The collector derives all eight ten-knot tables from accepted full100 results,
compares freshly computed sparse cases, and invokes the unchanged original table
exporter. Pc knots/entry/endpoint conventions must match full100 exports exactly.
Native import uses the existing repaired source-derived endpoint checker, with
no runtime modification. It is separate from scientific computation and preserves
completed output if import fails. This is not a 522-slot mapping test or reservoir
time-marching validation. Event-dominated non-fast-path coverage remains separate.

An isolated cache-bootstrap stage prepares strict loading. Four-core warm workers
reuse a process, release factors, collect garbage and retain per-core receipts.
Existing compilation depots are read-through dependencies; the first writable
depot is owned by this pilot. Compilation and compute timing are reported apart.

Build locally: python build.py; python -m unittest -v test_pilot.py.
Deploy: python build.py --deploy. Explicit bounded submission is remote.py submit.
All stage chains use afterok, advanced CPU account/QoS and verified allocations.
Submission intents persist before sbatch; an ambiguous return is held, not retried.
No full campaign or periodic automation is enabled.
