# Step62 secondary structural-fault trace

`step62_secondary_fault_trace.csv` defines the secondary structural fault used
in Fig. 1c. The repository does not contain a separate property-domain flag or
coordinate array for this fault. The trace is therefore reconstructed from the
actual Step62 mesh vertices at successive horizon offsets and extruded along
strike for visualization only.

The CSV coordinates are `(y, depth)` in metres. Its rows correspond to these
one-based nodes in `nodes_coordinates.dat`:

`10502, 9700, 10456, 10389, 1, 10537, 10114, 5676, 5675, 9753, 5678, 9752, 5679`.

The deep sequence `5676 -> 5675 -> 9753 -> 5678 -> 9752 -> 5679` is an exact
mesh-edge chain separating raw regions 56 and 57. The shallower nodes mark the
successive stratigraphic hinges. The renderer uses depth-monotone PCHIP
interpolation between these anchors. This surface represents structural
geometry, not a simulated uncertain fault-property region.
