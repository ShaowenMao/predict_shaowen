# Six MIP Pc-onset comparison runs

Isolated, same-draw comparison against the completed corrected-hysteresis MIP
property cases: low, intermediate and high risk, each at WRR30 and WRR60.

Only the PREDICT fault Pc pre-breakthrough convention changes. Each table keeps
its original saturation coordinates, MIP gas/brine Kr columns and post-breakthrough
Pc nodes exactly. Archived breakthrough pressure is placed at Sg=0 and connected
to the first retained post-breakthrough state. This is a table convention, not a
new hard invasion switch or reservoir-cell saturation clamp. No Pc/Kr upscaling
is rerun. The Pc-treatment label and external provenance identify the change.

All other physical inputs, solver tolerances and timestep controls remain those
of each matched MIP reference. Corrected runtime commit
c403d9dd1bbb424ddfae9ae659d900c26c84e59d keeps host hysteresis and disables it in
PREDICT regions 9:530. All simulations start at time zero; existing checkpoints,
production files, completed comparison results and running jobs are untouched.

All 210 report states and the same QoIs remain on project disk, not scratch.
Each full job depends on its own successful native assembled-model preflight.
Native table tests, exact input comparisons and a host/fault hysteresis audit
are mandatory. The copied final reporting check accepts the existing 210-state
retention policy; no simulation logic or scientific acceptance check is relaxed.

Slurm: mit_normal partition, mit_amf_advanced_cpu account and QoS. Preflights use
8 CPUs and 12 GiB (matched prior peaks 9.4-9.9 GiB); full runs retain 8 CPUs and
18 GiB from the matched references. Four hours for preflight and four days for
full runs. All six are independently submitted without a JobArrayTaskLimit.

Reproducibility: sources.json pins reference/donor files; input_manifest.json
and per-input pc_only_audit.json verify all 3132 tables; local_qualification.json
checks launcher and solver-setting equality. submissions.json is an incremental,
duplicate-protected ledger. Actual scheduling resources are checked after submission.
