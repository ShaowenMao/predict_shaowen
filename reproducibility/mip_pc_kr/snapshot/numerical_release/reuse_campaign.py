"""Qualification and migration into the exact-reuse dispatcher; old work is read-only."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import copy
import json
import os
import time
import reuse as R


@lru_cache(maxsize=1)
def method(C):
    names=('prepare.jl','worker.jl','Backend.jl','ManagedLU.jl','build_tables.py','native.jl','context.py','reuse.py')
    return C.digest(dict(schema=R.SCHEMA, kernels={n:C.sha(C.HERE/n) for n in names},
        external=C.S['external_files'], pipeline_hash=C.S['pipeline_hash'],
        scenario='WRR60_PHYSICAL_PREBREAKTHROUGH_FIXED100_DIP_DIRECT_2THREADS'))


def fingerprint(C, r):
    C.validate_replay(r)
    result=R.fingerprint(r['raw'],r['geometry'],r['slot'],r['raw_hash'],r['geometry_hash'],method(C),unplanned=True)
    # These immutable per-placement files are streamed once, not working arrays.
    # Release our read cache rather than letting page cache inflate job memory.
    if hasattr(os,'posix_fadvise'):
        for key in ('raw','geometry'):
            with Path(r[key]).open('rb') as stream:
                os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    return result


def parent_plan(C):
    root=Path(C.S['reuse_parent_root'])
    marker=C.read(root/'PLAN_ACCEPTED.json')
    assert marker['passed'] and marker['source_lock']==C.S['reuse_parent_source_lock']
    assert marker['sha256']==C.S['reuse_parent_plan_sha256']==C.sha(root/'PLAN.json')
    return root,C.read(root/'PLAN.json')


def verify_parent(C,root,key):
    d=C.read(root/'accepted'/(key+'.json'))
    assert d['complete'] and d['key']==key and d['native_passed'] and d['memory']['passed']
    assert d['source_lock']==C.S['reuse_parent_source_lock']
    for path,h in d['artifacts'].items(): assert C.sha(path)==h,path
    return d


def read_fingerprints(C,path,records,allow_incomplete_tail=False):
    found={}
    if not path.exists():return found
    with path.open() as stream:
        for line in stream:
            if allow_incomplete_tail and not line.endswith('\n'):
                C.event('PRESERVED_UNCOMMITTED_INDEX_TAIL',path=str(path))
                break
            d=json.loads(line)
            assert d['parent_plan']==C.S['reuse_parent_plan_sha256'] and d['method']==method(C)
            assert d['record']==C.digest(records[d['key']]) and d['key'] not in found
            found[d['key']]=d['fingerprint']
    return found


def start_scan(C):
    _,p=parent_plan(C)
    cases=sorted({r['case'] for r in p['records'].values()})
    jobs=[C.submit('reuse-index','index_'+case,1,2,3*3600,case) for case in cases]
    collector=C.submit('preflight','preflight',2,3,3600,deps=jobs)
    C.event('SHARDED_REUSE_INDEX_SUBMITTED',jobs=jobs,collector=collector,cases=cases)


def index_case(C,label,case):
    job=C.bound(label);C.pins(False);_,parent=parent_plan(C);records=parent['records']
    selected={k:r for k,r in records.items() if r['case']==case};assert selected
    path=C.ROOT/'index_shards'/(case+'.jsonl');path.parent.mkdir(exist_ok=True)
    done=path.with_suffix('.done.json');assert not done.exists()
    cached=read_fingerprints(C,C.ROOT/'fingerprints.jsonl',records,True)
    current=read_fingerprints(C,path,records)
    assert set(current)<=set(selected)
    cached.update(current)
    started=time.monotonic()
    with path.open('a') as out:
        for i,(key,r) in enumerate(sorted(selected.items()),1):
            if key not in current:
                fp=cached[key] if key in cached else fingerprint(C,r)
                d=dict(key=key,record=C.digest(r),fingerprint=fp,method=method(C),
                       parent_plan=C.S['reuse_parent_plan_sha256'])
                out.write(json.dumps(d,sort_keys=True)+'\n');out.flush()
            if i%25==0 or i==len(selected):
                os.fsync(out.fileno())
                C.event('REUSE_SHARD_PROGRESS',case=case,placements=i,total=len(selected),seconds=time.monotonic()-started)
    memory=C.context().memory_snapshot();assert memory['passed'] and memory['limit_gib']==job['memory_gib']
    C.write(done,dict(passed=True,case=case,count=len(selected),sha256=C.sha(path),
        source_lock=C.sha(C.HERE/'source_lock.json'),parent_plan=C.S['reuse_parent_plan_sha256'],
        job=os.environ['SLURM_JOB_ID'],memory=memory,reused_serial_fingerprints=len(set(cached)&set(selected))))


def preflight(C,label):
    job=C.bound(label);C.pins();root,old=parent_plan(C)
    assert not (C.ROOT/'PLAN_ACCEPTED.json').exists(), 'Preserve completed qualification'
    proof=C.S['reuse_parent_source_lock']
    parent_release=Path(C.S['reuse_parent_release'])
    assert C.sha(parent_release/'source_lock.json')==proof
    for n in ('prepare.jl','worker.jl','Backend.jl','ManagedLU.jl','MemoryProbe.jl',
              'context.py','build_tables.py','native.jl'):
        assert C.sha(C.HERE/n)==C.sha(parent_release/n),n
    records=old['records'];fps={}
    for case in sorted({r['case'] for r in records.values()}):
        journal=C.ROOT/'index_shards'/(case+'.jsonl');d=C.read(journal.with_suffix('.done.json'))
        assert d['passed'] and d['memory']['passed'] and d['source_lock']==C.sha(C.HERE/'source_lock.json')
        assert d['sha256']==C.sha(journal) and d['parent_plan']==C.S['reuse_parent_plan_sha256']
        part=read_fingerprints(C,journal,records)
        expected={k for k,r in records.items() if r['case']==case}
        assert set(part)==expected and d['count']==len(expected) and not set(part)&set(fps)
        fps.update(part)
    assert set(fps)==set(records)
    C.event('REUSE_INPUTS_VERIFIED',placements=len(fps),unique=len({v['numerical_key'] for v in fps.values()}))
    preferred=old['reused']
    owners,places,groups=R.group_records(records,fps,preferred)
    strata=defaultdict(list)
    for row in owners.values():strata[row['resource']['stratum']].append(row)
    strata={k:dict(memory_gib=v[0]['resource']['memory_gib'],canaries=C.canaries(v),count=len(v))
            for k,v in strata.items()}
    p=dict(records=owners,placements=places,groups=groups,strata=strata,held=old['held'],
        parent_root=str(root),parent_plan_sha256=C.S['reuse_parent_plan_sha256'],
        source_lock=C.sha(C.HERE/'source_lock.json'),scenario=old['scenario'],
        case_count=old['case_count'],reused=[],full_case_export_approved=False,
        all2430_expansion_approved_by_this_receipt=False)
    # Six completed controls retain their original receipts and scientific artifacts.
    for key in preferred:
        d=verify_parent(C,root,key);owner=places[key]['owner']
        assert d['replay_receipt_sha256']==places[key]['replay_receipt_sha256']
        receipt=root/'accepted'/(key+'.json')
        if owner!=key:continue
        new=copy.deepcopy(d)
        new.update(source_lock=p['source_lock'],numerical_key=places[key]['numerical_key'],
                   kind='EXACT_PARENT_RESULT_REUSE',parent_receipt=str(receipt),
                   parent_receipt_sha256=C.sha(receipt),scientific_recomputed=False)
        new['artifacts'][str(receipt)]=C.sha(receipt)
        if C.accepted_path(owner).exists():assert C.read(C.accepted_path(owner))==new
        else:C.write(C.accepted_path(owner),new)
        p['reused'].append(owner)
    duplicates=[r for r in owners.values() if len(r['placements'])>=2]
    assert duplicates,'No exact repeated cores found; do not claim qualification'
    pilot=min(duplicates,key=lambda r:(r['features']['num_cells'],
              r['features']['clay_cell_fraction'] not in (0,1),r['key']))
    p['reuse_pilot']=dict(owner=pilot['key'],alias=next(k for k in pilot['placements'] if k!=pilot['key']),
                         memory_gib=pilot['resource']['memory_gib'])
    if (C.ROOT/'PLAN.json').exists():assert C.read(C.ROOT/'PLAN.json')==p
    else:C.write(C.ROOT/'PLAN.json',p)
    memory=C.context().memory_snapshot()
    assert memory['passed'] and memory['limit_gib']==job['memory_gib']
    C.pins()
    C.write(C.ROOT/'PLAN_ACCEPTED.json',dict(passed=True,source_lock=p['source_lock'],
        sha256=C.sha(C.ROOT/'PLAN.json'),memory=memory,job=os.environ['SLURM_JOB_ID'],
        placements=len(places),unique_numerical_tasks=len(owners),duplicate_calculations_avoided=len(places)-len(owners),
        held=len(p['held']),controls_preserved=len(preferred),pilot=p['reuse_pilot'],
        numerical_dispatch_approved=False))
    C.event('REUSE_INDEX_PASS',placements=len(places),unique=len(owners),
            avoided=len(places)-len(owners),pilot=p['reuse_pilot'])
    launch(C)


def launch(C):
    p=C.plan();pilot=p['reuse_pilot']
    j=C.submit('reuse-pilot','reuse-pilot',2,pilot['memory_gib'],10800)
    controller=C.submit('controller','controller',1,2,8*3600,deps=[j])
    C.event('REUSE_QUALIFIED_CHAIN_SUBMITTED',pilot=j,controller=controller)


def verify_group(C,p,key):
    owner=p['records'][key];expected=p['groups'][owner['numerical_key']]['fingerprint']
    for k in owner['placements']:
        assert fingerprint(C,p['placements'][k])==expected,k


def pilot(C,label):
    j=C.bound(label);C.pins();p=C.plan();pair=p['reuse_pilot'];Ctxt=C.context()
    run=C.ROOT/'reuse_pilots'/os.environ['SLURM_JOB_ID']
    run.mkdir(parents=True,exist_ok=False)
    outputs=[]
    for which,key in enumerate((pair['owner'],pair['alias'])):
        r=p['placements'][key]
        assert j['memory_gib']==r['resource']['memory_gib']
        out=run/str(which)
        done=C.compute(r,out,j)
        assert done['native_passed'],'Both repeated-core exports must pass native tests'
        outputs.append((out,done))
    a,b=outputs
    fa=R.fingerprint(a[0]/'planned60/raw.bin',a[0]/'planned60/geometry.bin',
        p['placements'][pair['owner']]['slot'],C.sha(a[0]/'planned60/raw.bin'),
        C.sha(a[0]/'planned60/geometry.bin'),method(C))
    fb=R.fingerprint(b[0]/'planned60/raw.bin',b[0]/'planned60/geometry.bin',
        p['placements'][pair['alias']]['slot'],C.sha(b[0]/'planned60/raw.bin'),
        C.sha(b[0]/'planned60/geometry.bin'),method(C))
    assert fa==fb,'WRR60 planning depends on placement identity'
    import numpy as np
    arrays=[np.loadtxt(out/'managed/pass01/pc_kr_raw.csv',delimiter=',',skiprows=1) for out,_ in outputs]
    assert arrays[0].shape==arrays[1].shape==(100,8)
    assert np.array_equal(arrays[0],arrays[1]),'Repeated-core full curves not bitwise equal'
    assert C.sha(a[0]/'sgof.csv')==C.sha(b[0]/'sgof.csv'),'Export changed'
    summaries=[d['report']['summary'] for _,d in outputs]
    for field in ('axis','points','events','event_counts','owner_samples','fast_path',
                  'unique_mobility_states','intrinsic_conductance','intrinsic_permeability_m2'):
        assert summaries[0][field]==summaries[1][field],field
    verify_group(C,p,pair['owner'])
    receipt=C.accepted_path(pair['owner'])
    if receipt.exists():C.verify_accept(pair['owner'],True)
    else:C.write(receipt,a[1])
    memory=Ctxt.memory_snapshot();assert memory['passed']
    C.write(C.ROOT/'REUSE_QUALIFIED.json',dict(passed=True,job=os.environ['SLURM_JOB_ID'],
        source_lock=C.sha(C.HERE/'source_lock.json'),plan_sha256=C.sha(C.ROOT/'PLAN.json'),
        owner=pair['owner'],alias=pair['alias'],canonical_planned_inputs=fa,
        full_100_point_curves_bitwise_equal=True,export_tables_byte_identical=True,
        both_native_imports_passed=True,all_group_members_reverified=True,memory=memory,
        pilot_outputs=[dict(output=str(out),receipt=d) for out,d in outputs]))
    C.event('REUSE_QUALIFIED',owner=pair['owner'],alias=pair['alias'])


def require_qualified(C):
    d=C.read(C.ROOT/'REUSE_QUALIFIED.json')
    assert d['passed'] and d['source_lock']==C.sha(C.HERE/'source_lock.json')
    assert d['plan_sha256']==C.sha(C.ROOT/'PLAN.json')


def publish_mappings(C,p,accepted):
    for key,d in accepted.items():
        if key not in p['records'] or not d.get('native_passed'):continue
        target=C.ROOT/'mapped_groups'/(d['numerical_key']+'.json')
        if target.exists():continue
        assert d['source_lock']==C.sha(C.HERE/'source_lock.json') and d['memory']['passed']
        expected=p['records'][key]['numerical_key'];assert d['numerical_key']==expected
        # Includes inherited controls: validate every alias before publishing its pointer.
        verify_group(C,p,key)
        rows=[]
        for case in sorted({p['placements'][k]['case'] for k in p['records'][key]['placements']}):
            subset={'placements':{k:p['placements'][k] for k in p['records'][key]['placements']}}
            rows.extend(R.resolve_case(subset,case,{key:d},False))
        C.write(target,dict(complete=True,owner=key,numerical_key=expected,placements=rows,
            owner_receipt=str(C.accepted_path(key)),owner_receipt_sha256=C.sha(C.accepted_path(key)),
            source_lock=C.sha(C.HERE/'source_lock.json'),full_case_export_approved=False))
