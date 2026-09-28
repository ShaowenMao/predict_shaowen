# Exact upper-pressure endpoint recovery

Parent source: 075a961ccaaaed3bbd9165fa260591e12adfdd77f59b4304c50b55591b25a1a8.

Slot 295 stopped before MIP or phase solves. Its physical local endpoint was
18,375,591.899727955 Pa; the selected pressure was four Float64 steps lower,
18,375,591.89972794 Pa (relative shortfall 8.1092e-16). Several distinct
dictionary pressures had the same rounded logarithm. The unchanged nearest-log
lower-index tie rule consequently chose the lower endpoint neighbor.

This version changes only the pressure-plan boundary handling: when the selected
last pressure is below the local maximum endpoint, replace it with the already
present exact pmax. Require the shortfall to be at most 16 floating-point
increments at pmax; retain a hard failure for a larger shortfall. Do not relax
the physical coverage check, add a sample, change the pressure dictionary or
touch the other 99 selected pressures. Existing adequate plans remain unchanged.
No MIP, trapping, fine-property, TPFA, linear-solver, correction or export
implementation changes are made.

The frozen manifest and calculation keys remain input-lineage identifiers.
New results also record the actual implementation source and a distinct execution
key. Seven legacy accepted records may be retained only after source and output
hash verification and exact pressure/dictionary parity under the corrected plan.
They retain their original source and output receipts. New full-fault results
use this version and per-slot assembly provenance preserves both source versions.

Qualification runs only on a compute node: binary/endpoint edge tests, all eight
pilot pressure plans, unchanged Pc controls, integrated Pc/Kr controls and Python
orchestration tests. No successful pilot is rerun. Only slot 295 is retried,
reusing its hash-verified completed geometry preparation. A new afterok chain
then runs the pilot gate, remaining assignments and raw MAT assembly.

All original releases, initialization records, inputs, failed attempts and seven
completed results are preserved. No production code, results or jobs are changed.
The candidate raw export remains unapproved for reservoir production adoption.

Endpointfix2 retains endpointfix1's tested numerical change exactly. It limits
restart lineage checks to the frozen legacy receipts and the specific result
being consumed, rather than reading peer receipts while concurrent workers may
still be writing them. Qualification artifacts use recovery_endpointfix2;
endpointfix1 qualification results are retained. No core was rerun by fix1.
