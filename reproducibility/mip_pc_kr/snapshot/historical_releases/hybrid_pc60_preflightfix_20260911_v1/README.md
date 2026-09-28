# Intermediate WRR60 hybrid preflight recovery

The qualified hybrid specific MAT adds metadata.wrr_contact_angle_deg=60.
The inherited comparison audit did not recognize that intended addition.
No simulation or upscaling failure is repaired here, and no MAT file is changed.

The repaired audit explicitly verifies a scalar finite angle of 60 in both
candidate metadata and the qualified table export. It requires this metadata
to be absent in the production baseline. All existing protected-field checks,
522-table equality checks, geometry validation, and promotion guards remain.
Regression tests run within the native Engaging preflight, including negative
controls for wrong angles, K, porosity, geology, table indexing, and unrelated
metadata. Actual table checks follow those controls.

Run stage_recovery.py with Python 3.12 on Engaging to create reservoir_preflightfix1
under the intermediate WRR60 hybrid study. It refuses to overwrite an existing
root, verifies the original locked inputs, and seals the new launcher. The new
launcher preserves the full 8-CPU/18-GB settings, production solver settings,
fresh initialization, QoI schema 4, and all 210 report states on project storage.
The inherited preflight remains 8 CPUs/24 GB. Only job names and isolated output
paths change. Submit the existing source/submit_pair.sh from the new root;
the full run must depend on successful preflight via afterok.

Old failed preflight 22536389 and cancelled full job 22536390, all six completed
dynamic fits, and the qualified Pc/Kr exports are preserved unchanged.
