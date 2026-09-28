"""Native curve contract checks: flag unresolved export choices, never repair data."""
import bisect
import math


def audit(rows):
    assert len(rows)==100 and all(len(r)==8 for r in rows)
    assert all(math.isfinite(v) for r in rows for v in r)
    assert rows[0][0]==0 and all(b[0]>a[0] for a,b in zip(rows,rows[1:]))
    assert all(-1e-12<=r[1]<=1+1e-12 and all(-1e-8<=v<=1+1e-8 for v in r[2:4]) for r in rows)
    assert all(r[6] in (0,1) for r in rows)
    ds=[b[1]-a[1] for a,b in zip(rows,rows[1:])]
    repeats=[i for i,d in enumerate(ds) if d==0]
    near=[i for i,d in enumerate(ds) if 0<d<=1e-12]
    drops=[i for i,d in enumerate(ds) if d<0]
    # These changes should normally have the opposite sign.
    water_rise=max([b[2]-a[2] for a,b in zip(rows,rows[1:])]+[0.])
    gas_drop=max([a[3]-b[3] for a,b in zip(rows,rows[1:])]+[0.])
    flags=[]
    if repeats:flags.append('Pc_multivalued_at_repeated_Sg')
    if near:flags.append('nearly_repeated_Sg_review')
    if drops:flags.append('nonmonotone_Sg_review')
    if water_rise>1e-8 or gas_drop>1e-8:flags.append('nonmonotone_Kr_review')
    if not any(r[6] for r in rows):flags.append('no_gas_breakthrough')
    if rows[-1][2]>0:flags.append('positive_terminal_brine_Kr_do_not_force_zero')
    return dict(points=100,flags=flags,exact_repeated_sg_intervals=repeats,
        nearly_repeated_sg_intervals=near,sg_decrease_intervals=drops,
        max_water_kr_rise=water_rise,max_gas_kr_drop=gas_drop,
        sampled_max_sg=rows[-1][1],terminal_krw=rows[-1][2],terminal_krg=rows[-1][3],
        scalar_pc_export_ready=not(repeats or near or drops),
        reservoir_export_approved=False,raw_values_unchanged=True,
        interpolation_accuracy='not_measured_on_new_core',flags_are_job_failures=False)


def sample(rows,s,column,log=False):
    """No extrapolation and no choice between multiple Pc values at one Sg."""
    xs=[r[1] for r in rows]
    if any(b<a for a,b in zip(xs,xs[1:])):return None
    lo,hi=bisect.bisect_left(xs,s),bisect.bisect_right(xs,s)
    if hi>lo:
        vv=[r[column] for r in rows[lo:hi]]
        return vv[0] if all(v==vv[0] for v in vv) else None
    if lo==0 or lo==len(rows):return None
    a,b=rows[lo-1:lo+1]
    if b[1]-a[1]<=1e-12:return None
    w=(s-a[1])/(b[1]-a[1]);u,v=a[column],b[column]
    if log:
        if min(u,v)<=0:return None
        return u*math.exp(w*math.log1p((v-u)/u))
    return u+w*(v-u)


def compare(candidate,production_pc,production_kr):
    """Differences between methods, NOT validation errors or accuracy rankings."""
    common=[]
    for s in [i/100 for i in range(1,95)]:
        new=[sample(candidate,s,j,j==0) for j in (0,2,3)]
        old=[sample(production_pc,s,0,True),sample(production_kr,s,2),sample(production_kr,s,3)]
        common.append(dict(sg=s,candidate=new,production=old,
            pc_ratio=None if new[0] is None or old[0] in (None,0) else new[0]/old[0],
            delta_krw=None if None in (new[1],old[1]) else new[1]-old[1],
            delta_krg=None if None in (new[2],old[2]) else new[2]-old[2]))
    return dict(common_sg_comparisons=common,method_difference_not_numerical_error=True,
        gas_saturation_coordinate='physical_bulk_Sg',smoothing=False,extrapolation=False)
