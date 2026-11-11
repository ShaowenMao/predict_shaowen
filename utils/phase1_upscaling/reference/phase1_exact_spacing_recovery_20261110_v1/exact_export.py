"""Exact-node export: positive representable spacing, no physical-value edits.

The legacy positive 1e-12 spacing floor is replaced, not the curve science.
Every other legacy assertion remains. Native import with finite derivatives is
mandatory before publication; this module alone cannot accept a result.
"""
import hashlib
from pathlib import Path

OLD = '    assert np.all(np.diff(t[:, 0]) > 1e-12)'
NEW = '    assert np.all(np.diff(t[:, 0]) > 0), "Saturation nodes must be strictly ordered"'
POLICY = 'exact_positive_saturation_nodes_native_validated_v1'


def load_converter(path, expected_sha):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise RuntimeError('Frozen table converter source mismatch')
    source = data.decode('utf-8')
    if source.count(OLD) != 1:
        raise RuntimeError('Unexpected legacy spacing guard')
    source = source.replace(OLD, NEW)
    namespace = {'__name__': 'isolated_exact_node_converter', '__file__': str(path)}
    exec(compile(source, str(path) + ':exact-positive-spacing', 'exec'), namespace)
    return namespace['convert']
