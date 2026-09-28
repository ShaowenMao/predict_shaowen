"""Read old campaigns; write only the separately authorized Pc100/Kr10 campaign."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
ROOT=BASE/'campaigns/wrr60_pc100_kr10_2430_20260916_v1'
OLD=BASE/'campaigns/wrr60_phase1_2430_20260914_v1'
CAND=BASE/'candidates/wrr60_pc100_kr10_campaign_20260916_v1'
QUAL=CAND/'full_fault_dispatch_v2'
Q3=CAND/'qualification_v3'
WARM=BASE/'candidates/wrr60_phase1_production_candidate_20260912_v1/warm_worker_qualification/20260915_v2'
REPLAY=OLD/'releases/5748ac68fdc0be88'
NUM=OLD/'numerical_releases/6c240610c9f90e81'
PY=BASE/'candidates/wrr60_phase1_production_candidate_20260912_v1/reference_integration_v1/integration_v1/environments/export_py310_v1/bin/python'
JULIA='/orcd/software/core/001/pkg/julia/1.12.6/bin/julia'
ACCOUNT='mit_amf_advanced_cpu'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):
            h.update(b)
            if hasattr(os,'posix_fadvise'):os.posix_fadvise(f.fileno(),0,f.tell(),os.POSIX_FADV_DONTNEED)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x,replace=False):
    p=Path(p);assert p.resolve().is_relative_to(ROOT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:
        assert read(p)==x,'Refuse overwrite: '+str(p);return
    t=p.with_name(p.name+'.partial.'+str(os.getpid()))
    with t.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if replace:os.replace(t,p)
    else:os.link(t,p);t.unlink()
def pins_ok(d):
    for p,h in d.items():assert sha(p)==h,'Changed pinned file: '+str(p)
def verify():
    d=read(HERE/'LOCK.json');pins_ok({HERE/n:h for n,h in d['files'].items()});pins_ok(d['external'])
    assert d['authorized_full_campaign'] and not d['reservoir_submission']
    return d
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m
def resource():return module(QUAL/'resource.py','launch_resource')
def memory(mib):
    R=resource();m=R.snapshot(R.cgroup(os.environ['SLURM_JOB_ID']))
    assert R.passed(m,mib),m
    return m
def allocation(cpus,mib):
    s=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
    resource().parse_allocation(s,cpus,mib);return s
def event(name,**kw):print(json.dumps(dict(event=name,**kw)),flush=True)
def run(args,log,env=None):
    with Path(log).open('x') as f:subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)

