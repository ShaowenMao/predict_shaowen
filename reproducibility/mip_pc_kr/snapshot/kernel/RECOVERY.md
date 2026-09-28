# Preserve Completed Numerical Work

Three v2 workers completed their first sparse curve before an overly strict CSV
text-equality review stopped the batch. Across all 100 states, microscopic state
hashes, pressure values, event counts and gas-breakthrough flags were identical
to the stored full100 controls. All selected Kr values were bitwise identical.
Only scalar pore-volume averages differed, at most 5.551115123125783e-16 absolute
and eight ULP in the inspected outputs. The unchanged observe() implementation
uses array sums. The benchmark and new launcher differ in their explicit runtime
CPU target, so cross-executable bitwise scalar reduction agreement is not assumed.

V3 does not change the numerical kernel or launch flags. It keeps microscopic
state hashes and discrete events exact, separately bounds only scalar average
differences by 32 machine epsilons relative, and requires identical selected state
indices. Zero remains exact. A 1e-12 change, altered hash, event, or pressure is
rejected by regression tests. Existing corrected-solver accuracy gates remain.

The three completed artifacts are hash-pinned in RECOVERY_INPUTS.json and copied
read-only into the new validation attempt. Their MIP and flow solves are not
repeated. Warm runs continue through the other batch members; their measured
memory includes consecutive live computations, not only artifact recovery.
The old source directories, failed checks and successful cache preparation are
preserved. No production or old campaign source/result is changed.
