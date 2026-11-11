"""Private immutable records, per-packet locking and compact result integrity."""
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import uuid
import zipfile

CODE=Path(__file__).resolve().parent
ROOT=Path(os.environ.get('PHASE1_RUN_ROOT',str(CODE/'output')))

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:
        require(read(p)==v,'Immutable record conflict: '+str(p));return
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('x') as f:
        json.dump(v,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    try:
        if replace: os.replace(tmp,p)
        else: os.link(tmp,p)
    finally:
        if tmp.exists():tmp.unlink()

@contextlib.contextmanager
def locked(p):
    import fcntl
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        yield f.fileno()

def key(r):
    require(re.fullmatch(r's\d{2}_c\d{3}_pc0[46]_ca40_IND_00[1-8]',r['case_id']) is not None,'Invalid new case')
    require(type(r['slot']) is int and 1<=r['slot']<=522,'Invalid core')
    return r['case_id']+f"_core{r['slot']:03d}"

def verify_source():
    lock=read(CODE/'LOCK.json')
    for p,h in lock['code'].items():require(sha(CODE/p)==h,'Runner source changed: '+p)
    return lock

def verify_archive(p,r,method):
    with zipfile.ZipFile(p) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate archive members')
        proof=json.loads(z.read('PASS.json'))
        require(proof['record_sha256']==digest(r) and proof['key']==key(r) and proof['method']==method,'Wrong result identity')
        require(proof['scientific_pass'] and not proof.get('fixture',False),'Not a scientific result')
        require(set(names)==set(proof['files'])|{'PASS.json'},'Incomplete archive')
        for n,h in proof['files'].items():
            rel=PurePosixPath(n)
            require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in n,'Unsafe member')
            require(hashlib.sha256(z.read(n)).hexdigest()==h,'Corrupt result member')
    return proof

def publish(work,r,method,target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists(),'Existing result cannot be replaced')
    files={p.relative_to(work).as_posix():sha(p) for p in sorted(work.rglob('*')) if p.is_file()}
    proof=dict(key=key(r),record_sha256=digest(r),method=method,scientific_pass=True,fixture=False,files=files)
    tmp=target.with_name(target.name+'.'+uuid.uuid4().hex+'.partial')
    with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n in files:z.write(work/n,n)
        z.writestr('PASS.json',json.dumps(proof,sort_keys=True))
    verify_archive(tmp,r,method)
    with tmp.open('r+b') as f:os.fsync(f.fileno())
    os.link(tmp,target);tmp.unlink()
    return archive_receipt(target,r,method)

def archive_receipt(target,r,method):
    target=Path(target);verify_archive(target,r,method)
    return dict(key=key(r),record_sha256=digest(r),method=method,archive_sha256=sha(target),bytes=target.stat().st_size,scientific_pass=True)

def action(work,result,r,method):
    if result.exists():verify_archive(result,r,method);return 'DONE'
    if (work/'POSTFLOW_HOLD.json').exists():return 'HOLD_POSTFLOW'
    if (work/'flow/CORE_PASS.toml').exists():return 'FINISH'
    if (work/'SOLVE_STARTED.toml').exists():return 'HOLD_UNCERTAIN'
    if (work/'FLOW_ERROR.toml').exists():return 'HOLD_PREPARATION'
    if (work/'HOLD.json').exists():return 'HOLD_PREPARATION'
    return 'PREPARE'
