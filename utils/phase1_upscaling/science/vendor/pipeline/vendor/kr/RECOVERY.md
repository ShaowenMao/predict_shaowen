# Input-export recovery

Original pilot 22389268 failed before flow benchmarking because MATLAB's Java
MessageDigest.update overload rejects an empty array. Empty phase networks are
valid zero-flow states, not numerical failures. The dependent core 3/18 array
22389275 was automatically cancelled without running.

This release skips zero-length hash updates, which is exactly equivalent to
hashing an empty byte stream. Four MATLAB regression checks cover an empty
network, a one-node spanning network with no internal edges, and unchanged
nonempty byte encoding. The check also runs before each real-core export.

Only export_packet.m and the two new fingerprint helpers differ from the original
numerical/workflow source. All phase-flow code, C++ adapter/kernels, solvers,
tolerances, sampling, tests and orchestration remain unchanged. Original source
and failed results are retained. There are no completed flow comparisons to
repeat. The recovered jobs use a new output root and source receipt.
