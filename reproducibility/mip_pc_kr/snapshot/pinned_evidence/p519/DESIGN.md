# Exact Core 518 Managed-Only Tight-Allocation Canary

The completed 32 GiB candidate job 22739109 passed the original 100-state
numerical checks, with a 16.1558036804 GiB job peak and zero memory-limit events.
The unchanged sizing rule requires 21.0025447845 GiB, rounded up to 22 GiB.
This canary tests that estimate rather than carrying a 32 GiB allowance forward.

Only this exact s05_c002_case01 Core 518 WRR60 realization is repeated, with
two CPUs, 22 GiB and one hour under the advanced account AND QoS. It uses a new
private release/output directory on project disk. The baseline is copied from
completed job 22737254, never rerun. The same exact prepared geometry, raw input,
Julia flow worker, MIP algorithm, source pins, correction checks and export are
used. The run wrapper differs from the qualified 32 GiB wrapper ONLY in its two
resource assertions. No other core memory requests or production files change.

Acceptance requires full numerical and export checks plus the unchanged
max(peak + 1 GiB, 1.3 * peak) headroom gate and zero memory-limit events in this
fresh job. Failure preserves evidence; it must not trigger a blanket allocation
increase or automatic escalating retries. The successful 32 GiB qualification
remains valid for its own measured context. Full campaign/dispatcher admission
and multi-core batch memory are not qualified by this isolated canary.
