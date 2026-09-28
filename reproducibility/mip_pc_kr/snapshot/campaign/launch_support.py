"""Frozen launch contract, bounded identity verification and scheduler admission."""
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import re
import shutil
import sqlite3
import time
from functools import lru_cache
from common_launch import *

OLD_METHOD='645ab862dcc1f410c51b615aa2291ecf8e487c713fde9fd16b0f07791443eff1'
CLASSES={'mixed_60000':(2816,1024),'mixed_90000':(7040,1024),
         'mixed_120000':(13952,1152),'mixed_150000':(20096,1280),
         'clay_120000':(3712,1152)}

def guard():
    d=read(HERE/'LAUNCH_LOCK.json')
    assert d['base_lock_sha256']==sha(HERE/'LOCK.json')
    pins_ok({HERE/n:h for n,h in d['files'].items()});pins_ok(d['external'])
    assert d['contract']['kr_states']==10 and d['contract']['clay_angle']==60
    return d

@lru_cache(maxsize=1)
def method_hash():return read(HERE/'LAUNCH_LOCK.json')['method_sha256']

def method_key(old):return digest(dict(input_identity=old,method=method_hash()))

def connection():
    p=ROOT/'state/dispatch.sqlite'
    db=sqlite3.connect(p,timeout=60)
    db.execute('PRAGMA busy_timeout=60000');db.execute('PRAGMA cache_size=-16384')
    return db

def memory_mib(value):
    m=re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)',value)
    if not m:raise ValueError('Unrecognized scheduler memory: '+value)
    factor={'':1,'K':Decimal(1)/1024,'M':1,'G':1024,'T':1024**2}[m[2]]
    return int((Decimal(m[1])*factor).to_integral_value(rounding=ROUND_CEILING))

def admission(rows,own,cpus,mib):
    used_cpu=used_mem=0;seen=set();deferred=[]
    for job,state,cpu,mem,reason,account,qos in rows:
        assert job not in seen;seen.add(job)
        if account!=ACCOUNT:continue
        assert qos==ACCOUNT
        assert state in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED')
        if state=='PENDING' and job not in own and reason in ('Dependency','JobHeldUser','DependencyNeverSatisfied'):
            deferred.append(job);continue
        used_cpu+=int(cpu);used_mem+=memory_mib(mem)
    assert min(cpus,mib,used_cpu,used_mem)>=0
    return dict(passed=used_cpu+cpus<=500 and used_mem+mib<=1984*1024,
                used_cpus=used_cpu,used_mib=used_mem,add_cpus=cpus,add_mib=mib,
                deferred_external=deferred)

def queue():
    s=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],
        '-o','%i|%T|%C|%m|%r|%a|%q'],text=True,timeout=30)
    rows=[x.strip().split('|') for x in s.splitlines() if x.strip()]
    assert all(len(x)==7 for x in rows);return rows

@contextmanager
def single_writer():
    import fcntl
    p=ROOT/'state/controller.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def storage_probe(tag):
    p=ROOT/'audit'/('storage_'+tag+'.bin');b=os.urandom(1048576);start=time.monotonic()
    with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    with p.open('rb') as f:assert f.read()==b
    free=shutil.disk_usage(ROOT).free
    assert free>=2*1024**4,'Less than 2 TiB filesystem free; admission held'
    return dict(passed=True,seconds=time.monotonic()-start,free_bytes=free,
                file=str(p),sha256=sha(p),quota_not_certified=True)

def route(row):
    f=row['features'];n=f['num_cells'];dims=f['cartesian_dimensions']
    if len(dims)!=3 or dims[0]!=100 or dims[2]!=100 or n!=dims[0]*dims[1]*dims[2]:
        return None,'HOLD_UNQUALIFIED_GEOMETRY'
    if max(f['log10_permeability_span'])>8.135716210401004:
        return None,'HOLD_UNQUALIFIED_CONTRAST'
    if f['clay_cell_fraction']==1 and n<=120000:return 'clay_120000',None
    for band in (60000,90000,120000,150000):
        if n<=band:return 'mixed_'+str(band),None
    return None,'HOLD_UNQUALIFIED_SIZE'

def canonical_equal(a,b,geometry=False,raw_normal=None):
    n=104 if geometry else 96
    with Path(a).open('rb') as x,Path(b).open('rb') as y:
        h1,h2=bytearray(x.read(n)),bytearray(y.read(n));h1[8:16]=h2[8:16]=bytes(8)
        if geometry:h1[72:104]=h2[72:104]=bytes.fromhex(raw_normal)
        assert h1==h2
        while True:
            aa,bb=x.read(1048576),y.read(1048576);assert aa==bb,'Canonical byte mismatch'
            for f in (x,y):
                if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,f.tell(),os.POSIX_FADV_DONTNEED)
            if not aa:break

def verify_owner(item):
    reuse=module(QUAL/'reuse.py','launch_reuse')
    bounded=module(QUAL/'bounded_io.py','launch_bounded');reuse.normalized_file=bounded.normalized_file
    first=None
    for row in [item['row'],*item['aliases']]:
        assert row['numerical_key']==item['old_numerical_key']
        fp=reuse.fingerprint(row['raw'],row['geometry'],row['slot'],row['raw_hash'],row['geometry_hash'],OLD_METHOD,unplanned=True)
        assert fp['numerical_key']==item['old_numerical_key']
        if first:
            canonical_equal(row['raw'],first['raw'])
            canonical_equal(row['geometry'],first['geometry'],True,fp['raw_scientific_sha256'])
        else:first=row
    return dict(passed=True,old_numerical_key=item['old_numerical_key'],
                exact_aliases_verified=1+len(item['aliases']))

def toml_config(p,top,cases=(),pins=None):
    text=''.join(k+' = '+json.dumps(v)+'\n' for k,v in top.items())
    for c in cases:text+='\n[[cases]]\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in c.items())
    if pins is not None:text+='\n[pins]\n'+''.join(json.dumps(str(k))+' = '+json.dumps(v)+'\n' for k,v in pins.items())
    with Path(p).open('x') as f:f.write(text)

def julia_env(batch):
    m=read(Q3/'manifest.json');depot=batch/'depot';depot.mkdir(exist_ok=True)
    return dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',
        JULIA_DEPOT_PATH=':'.join(map(str,[depot,ROOT/'cache/depot',Q3/'cache/depot',m['depot']])),
        JULIA_NUM_THREADS='2',JULIA_NUM_PRECOMPILE_TASKS='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')

def julia_args(script,cfg,threads=2):
    m=read(Q3/'manifest.json')
    return [JULIA,'--startup-file=no','--math-mode=ieee','--check-bounds=yes','--threads='+str(threads),
        '--gcthreads=1','--cpu-target=generic','--project='+m['project'],
        '--compiled-modules=strict','--pkgimages=existing',str(script),str(cfg)]

def receipt_path(batch,stage):return ROOT/'batches'/batch/'receipts'/(stage+'.json')

def stage_receipt(batch,stage):
    p=receipt_path(batch,stage);r=read(p)
    accepted={sha(HERE/'LAUNCH_LOCK.json'),*read(HERE/'LAUNCH_LOCK.json').get('inherited_launch_locks',[])}
    assert r['passed'] and r['lock_sha256'] in accepted
    pins_ok(r['artifacts']);return r

def concurrency(passed):return 1 if passed==0 else 2 if passed<3 else 8 if passed<7 else 32 if passed<15 else 250
