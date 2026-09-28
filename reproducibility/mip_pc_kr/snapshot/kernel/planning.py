"""Pure design checks. No scheduler, simulation, production writes or network I/O."""
import hashlib
import json
import math

MIB = 1024 ** 2
IDENTITY_FIELDS = (
    "fine_input_sha256", "geometry_sha256", "pc_physics_sha256",
    "pressure_plan_sha256", "flow_physics_sha256", "solver_release_sha256",
    "normalization_sha256",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def select_states(values, budget=10):
    """Independent Python specification of the tested ordered Julia selector."""
    s = [float(x) for x in values]
    if budget < 2 or len(s) < 2 or not all(math.isfinite(x) for x in s):
        raise ValueError("Invalid selection input")
    if any(a > b for a, b in zip(s, s[1:])):
        raise ValueError("Nonmonotone MIP saturation")
    ids = [0] + [i for i in range(1, len(s)) if s[i] > s[i-1]]
    if len(ids) == 1:
        return [0, len(s)-1], [s[0], s[-1]]
    ids[-1] = len(s)-1
    count, n = min(budget, len(ids)), len(ids)
    span = s[-1] - s[0]
    targets = [s[0] + span * j / (count-1) for j in range(count)]
    costs = [[math.inf] * n for _ in range(count)]
    parent = [[-1] * n for _ in range(count)]
    costs[0][0] = 0.
    for j in range(1, count):
        for i in range(j, n-count+j+1):
            if j == count-1 and i != n-1:
                continue
            for k in range(j-1, i):
                v = costs[j-1][k] + ((s[ids[i]]-targets[j])/span)**2
                if v < costs[j][i]:
                    costs[j][i], parent[j][i] = v, k
    pos = [0] * count
    pos[-1] = n-1
    for j in range(count-1, 0, -1):
        pos[j-1] = parent[j][pos[j]]
    return [ids[i] for i in pos], targets


def compatible(old, desired):
    """A digest lookup alone never qualifies an old source for read-through."""
    for k in IDENTITY_FIELDS:
        value = desired.get(k)
        if not isinstance(value, str) or len(value) != 64:
            return False
        if any(c not in "0123456789abcdef" for c in value):
            return False
        if old.get(k) != value:
            return False
    return True


def artifact_route(receipt, desired):
    if receipt is None:
        return "REPLAY_AND_SPARSE_COMPUTE"
    if not receipt.get("verified") or not compatible(receipt.get("identity", {}), desired):
        return "HOLD_UNVERIFIED_OR_INCOMPATIBLE"
    if receipt.get("raw_pc100_accepted") and receipt.get("raw_kr100_accepted"):
        return "DERIVE_TEN_VALUES_NO_FLOW"
    if receipt.get("raw_pc100_accepted"):
        return "REUSE_PC_REPLAY_SELECTED_FLOW"
    return "USE_FINE_INPUTS_TWO_PASS_SPARSE"


def sparse_key(identity, selector_sha256, ids, sg, pressures):
    if not compatible(identity, identity):
        raise ValueError("Incomplete scientific identity")
    if len(ids) != len(sg) or len(ids) != len(pressures) or len(ids) > 10:
        raise ValueError("Invalid sparse plan")
    return digest(dict(schema="pc100_kr10_v1", identity=identity,
                       selector_sha256=selector_sha256, sample_ids=ids,
                       actual_sg=sg, pressure_pa=pressures,
                       interpolation="linear_in_sg", budget=10))


def memory_request_mib(peaks_bytes, qualified_floor_mib=0):
    if not peaks_bytes or any(type(v) is not int or v <= 0 for v in peaks_bytes):
        raise ValueError("Uncensored whole-job peaks required")
    if type(qualified_floor_mib) is not int or qualified_floor_mib < 0:
        raise ValueError("Invalid validated allocation floor")
    required = max((6*max(peaks_bytes)+4)//5, qualified_floor_mib*MIB)
    return ((required+128*MIB-1)//(128*MIB))*128


def completed_memory_ok(peak_bytes, allocation_mib, limit_events):
    return (peak_bytes > 0 and limit_events == 0
            and 21*peak_bytes <= 20*allocation_mib*MIB)


def fits(used_cpus, used_mib, add_cpus, add_mib, cap_cpus=500, cap_mib=1984*1024):
    if min(used_cpus, used_mib, add_cpus, add_mib) < 0:
        raise ValueError("Negative reservation")
    return used_cpus+add_cpus <= cap_cpus and used_mib+add_mib <= cap_mib


def paired_grid(pc_rows, kr_rows):
    """Preserve all Pc samples and evaluate only the approved sparse interpolant.

    This is a pre-export prototype, not an SGOF endpoint/entry policy.
    """
    import numpy as np
    s = np.asarray([r["sg_physical"] for r in pc_rows], dtype=float)
    p = np.asarray([r["pressure_pa"] for r in pc_rows], dtype=float)
    x = np.asarray([r["sg_physical"] for r in kr_rows], dtype=float)
    phases = np.asarray([[r["krw"], r["krg"]] for r in kr_rows], dtype=float)
    if not all(np.isfinite(v).all() for v in (s, p, x, phases)):
        raise ValueError("Nonfinite raw values")
    if len(s) != 100 or len(x) < 2 or len(x) > 10 or np.any(np.diff(s) < 0):
        raise ValueError("Unexpected state count or ordering")
    if np.any(np.diff(p) <= 0) or x[0] != s[0] or x[-1] != s[-1]:
        raise ValueError("Pressure plan or endpoint mismatch")
    if np.any(phases < 0) or np.any(phases > 1+1e-5):
        raise ValueError("Invalid raw relative permeability")
    ids = [int(r["sample"])-1 for r in kr_rows]
    if len(set(ids)) != len(ids) or ids != sorted(ids) or ids[0] != 0 or ids[-1] != 99:
        raise ValueError("Sparse indices do not retain the original endpoints")
    for i, row in zip(ids, kr_rows):
        if not 0 <= i < len(s) or float(row["sg_physical"]) != s[i] or float(row["pressure_pa"]) != p[i]:
            raise ValueError("Not an actual saved MIP state")
    if x[0] == x[-1]:
        if np.any(phases != phases[0]):
            raise ValueError("Conflicting phase flow at constant saturation")
        values = np.repeat(phases[:1], len(s), axis=0)
    else:
        if np.any(np.diff(x) <= 0):
            raise ValueError("Repeated interior Kr saturation")
        values = np.column_stack([np.interp(s, x, phases[:, j]) for j in (0, 1)])
    return np.column_stack((p, s, values))
