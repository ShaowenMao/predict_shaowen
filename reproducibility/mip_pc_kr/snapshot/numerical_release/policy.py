"""Admission policy and pure scheduler-state helpers; no scientific approximations."""
import math

QOS='mit_amf_advanced_cpu'

def route(f,contrast_limit):
    n=f['num_cells'];dims=f['cartesian_dimensions'];c=f['clay_cell_fraction']
    if len(dims)!=3 or dims[0]!=100 or dims[2]!=100 or math.prod(dims)!=n:
        return None,'unsupported_geometry'
    if not 0<=c<=1 or not 0<n<=150000: return None,'outside_resource_envelope'
    if any(not math.isfinite(v) or v>contrast_limit+1e-12 for v in f['log10_permeability_span']):
        return None,'outside_contrast_envelope'
    homogeneous=c in (0,1)
    band=next(b for b in (30000,60000,90000,120000,150000) if n<=b)
    i=(30000,60000,90000,120000,150000).index(band)
    mem=((4,4,6,8,12) if homogeneous else (4,6,10,16,24))[i]
    material='clay' if c==1 else 'sand' if c==0 else 'mixed'
    return dict(stratum=f'{material}_{band}',memory_gib=mem,cpus=2,band=band),None

def canaries(records):
    ordered=sorted(records,key=lambda r:(-r['features']['num_cells'],
        -max(r['features']['log10_permeability_span']),r['key']))
    if not ordered:return []
    first=ordered[0]
    rest=[r for r in ordered[1:] if r['key']!=first['key']]
    if not rest:return [first['key']]
    second=max(rest,key=lambda r:(abs(r['features']['clay_cell_fraction']-first['features']['clay_cell_fraction']),
                                 r['features']['num_cells'],r['key']))
    return [first['key'],second['key']]

def room(cpus,mem,need_cpu,need_mem): return cpus+need_cpu<=500 and mem+need_mem<=1984

def claim_action(claim,found=None):
    if claim is None:return 'SUBMIT'
    if claim.get('job'):return 'PRESERVE'
    if found:return 'RECONCILE'
    return 'HOLD_UNCERTAIN'

def gate_status(keys,accepted,failed):
    if any(k in failed for k in keys):return 'HOLD'
    if any(k in accepted and (not accepted[k].get('native_passed') or not accepted[k]['memory']['passed']) for k in keys):return 'HOLD'
    if all(k in accepted and accepted[k].get('native_passed') and accepted[k]['memory']['passed'] for k in keys):return 'PASS'
    return 'WAIT'

def seconds_allowed(now,cutoff,requested=10800):
    return max(0,min(requested,int(cutoff-now-1800)//60*60))
