"""Isolated, immutable 522-assignment rollout on a completed production case."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from export_audit import audit,compare

ROOT=Path(__file__).resolve().parent
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
OUTPUT_BASE=BASE/'candidates/cell_pc_mip_kr_v1_20260908'
PROD=BASE/'production_runs/independent_full_fault_v1_phase1_20260809_v1'
PHYSICS=BASE/'orchestration/independent_full_fault_v1_04927f5/physics_repo'
DATA=BASE/'production_freezes/collapsed_cell_union_20260722_v7/inputs/predict'
MRST=Path('/orcd/pool/006/shaowen/predict_shaowen/software/mrst-2025a/SINTEF-AppliedCompSci-MRST-75749fa')
PHYSICS_COMMIT='68351e35f3679317b35532a9ca0533674e0aafb5'
PIPELINE_HASH='d6de18a5bb2428fde69846211c51e7fc535f28793c62d3b1e2985d4f88d46a8a'
CASE='s03_c022_case12'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def quote(x):return "'"+str(x).replace("'","''")+"'"
def rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def numeric(path,header=False):
    with Path(path).open() as f:
        r=csv.reader(f)
        if header:next(r)
        return [[float(x) for x in z] for z in r]
def csvwrite(path,values):
    with Path(path).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
def hashes(root):return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
def check_hashes(root,values):
    for p,h in values.items():assert sha(root/p)==h,('Input/output changed',root/p)
def source_files():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'source_lock.json'}
def verify():
    d=read(ROOT/'source_lock.json');found=source_files();assert d['files']==found and digest(found)==d['sha256']
    p=ROOT/'vendor/pipeline';lock=read(p/'source_lock.json');assert lock['sha256']==PIPELINE_HASH
    check_hashes(p,lock['files']);return d['sha256']
def scope(out):
    out=out.resolve();assert OUTPUT_BASE in out.parents and ROOT not in out.parents and out!=ROOT
    return out
def coverage(values):
    assert len(values)==522
    seen={(x['window'],int(x['slice_index'])) for x in values}
    assert seen=={(f'famp{w}',s) for w in range(1,7) for s in range(1,88)}
    assert all(x['geology_id']=='s03_c022' and x['case_id']=='12' for x in values)
    return sorted(values,key=lambda x:(int(x['window'][4:]),int(x['slice_index'])))


def prepare(out):
    out=scope(out);assert not out.exists();source=verify()
    values=coverage([x for x in rows(PROD/'case_work_manifest/geology_assignments/s03_c022.csv') if x['case_id']=='12'])
    cin=PROD/'case_inputs/cases/s03_c022/case12';cres=PROD/'case_results/cases/s03_c022/case12'
    a=read(cin/'case_inputs.done.json');b=read(cres/'case.done.json')
    assert a['status']=='assembled' and b['status']=='complete'
    assert b['pc_assignment_count']==522 and b['dynamic_kr_representative_count']==6
    assert b['physics_commit']==PHYSICS_COMMIT
    summary_name=f'pc_curve_summary_{CASE}_ip_full87.csv'
    summary_file=cin/'inputs'/summary_name
    assert sha(summary_file)==a['files'][summary_name]['sha256']
    summaries={(x['Window'],int(x['SliceIndex'])):x for x in rows(summary_file)}
    assert len(summaries)==522
    baseline_mat=cres/f'kr/reservoir_ready/fault_properties_{CASE}_pc_full_slice.mat'
    sources={str(cin/'case_inputs.done.json'):sha(cin/'case_inputs.done.json'),
        str(cres/'case.done.json'):sha(cres/'case.done.json'),str(summary_file):sha(summary_file),
        str(PROD/'case_work_manifest/geology_assignments/s03_c022.csv'):sha(PROD/'case_work_manifest/geology_assignments/s03_c022.csv'),
        str(baseline_mat):b['files'][baseline_mat.relative_to(cres).as_posix()]['sha256']}
    native=cin/f'inputs/pc_native_curve_points_{CASE}_ip_full87.csv'
    sources[str(native)]=a['files'][native.name]['sha256']
    (out/'inputs').mkdir(parents=True);csvwrite(out/'inputs/assignments.csv',values)
    records=[]
    for slot,x in enumerate(values,1):
        y=summaries[(x['window'],int(x['slice_index']))]
        assert x['task_id']==y['TaskId'] and x['selected_sample_index']==y['SelectedSampleIndex'] and x['exact_replay_seed']==y['ReplaySeed']
        checkpoint=DATA/x['checkpoint_relative_path'];assert checkpoint.exists()
        folder=out/'inputs'/f'slot{slot:03d}';folder.mkdir();csvwrite(folder/'selection.csv',[x])
        r=dict(slot=slot,assignment=x,production_summary=y,checkpoint=str(checkpoint),checkpoint_sha256=x['checkpoint_sha256'])
        r['calculation_key']=digest(dict(source=source,record=r))
        write(folder/'identity.json',r);records.append(r)
    pilots=[(w-1)*87+int(a['swi_medoid_slices'][f'famp{w}']) for w in range(1,7)]
    pilots+= [min(range(1,523),key=lambda i:float(values[i-1]['log_kzz'])),max(range(1,523),key=lambda i:float(values[i-1]['log_kzz']))]
    pilots=sorted(set(pilots))
    write(out/'inputs/manifest.json',dict(source=source,case=CASE,records=records,pilots=pilots,
        remaining=[i for i in range(1,523) if i not in pilots],sources=sources,baseline_mat=str(baseline_mat),
        native_pc=str(native),production_method_hash=b['method_config_sha256'],
        assignments=522,unique_replay_identities=len({x['task_id'] for x in values}),
        cross_assignment_curve_reuse=False,production_promotion=False,
        selection_reason='Completed intermediate-risk case, variable Kzz, original six dynamic-Kr representatives and reservoir QoIs available'))
    write(out/'inputs_lock.json',hashes(out/'inputs'))
    print('PREPARED',CASE,'pilots',pilots,'assignments',len(records),flush=True)


def command(folder,args):
    folder.mkdir(parents=True,exist_ok=False);start=time.monotonic();print('STAGE',folder.name,flush=True)
    with (folder/'process.log').open('x') as f:
        p=subprocess.run(['/usr/bin/time','-v','-o',str(folder/'resources.txt')]+args,cwd=folder,stdout=f,stderr=subprocess.STDOUT,timeout=21000)
    write(folder/'receipt.json',dict(args=args,returncode=p.returncode,seconds=time.monotonic()-start))
    assert p.returncode==0,str(folder)+' failed'


def matlab(folder,expression):command(folder,['matlab','-batch',f'addpath({quote(ROOT)});'+expression])


def external_inventory():
    found={}
    for root in [PHYSICS,MRST]:
        for parent,dirs,names in os.walk(root):
            dirs[:]=[x for x in dirs if x not in ['.git','data','results','__pycache__']]
            for name in names:
                p=Path(parent)/name
                if p.suffix in ['.m','.mexa64','.so']:found[str(p)]=sha(p)
    assert subprocess.check_output(['git','-C',str(PHYSICS),'rev-parse','HEAD'],text=True).strip()==PHYSICS_COMMIT
    return found


def initialize(out,m):
    assert not (out/'initialized.json').exists()
    check_hashes(Path('/'),m['sources'])
    assert shutil.disk_usage(out).free>150*1024**3,'Need 150 GiB project-space headroom before 522-core pilot'
    inventory=external_inventory();write(out/'external_inventory.json',inventory)
    baseline=out/'baseline';baseline.mkdir()
    shutil.copy2(m['baseline_mat'],baseline/'production_fault_properties.mat')
    assert sha(baseline/'production_fault_properties.mat')==m['sources'][m['baseline_mat']]
    native=rows(m['native_pc']);by_slot={}
    for x in native:by_slot.setdefault((x['Window'],int(x['SliceIndex'])),[]).append(float(x['PcPa']))
    for r in m['records']:
        x=r['assignment'];b=r['production_summary'];folder=baseline/f'slot{r["slot"]:03d}';folder.mkdir()
        write(folder/'summary.json',dict(slot=r['slot'],NumCells=int(b['NumCells']),PoreVolume=float(b['PoreVolume']),
            BulkVolume=float(b['BulkVolume']),production_pressure_pa=sorted(set(by_slot[(x['window'],int(x['slice_index']))]))))
    matlab(out/'baseline_export_process',f'baseline_export({quote(baseline/"production_fault_properties.mat")},{quote(out/"inputs/assignments.csv")},{quote(baseline/"curves")});')
    matlab(out/'raw_export_test_process',f'test_raw_export({quote(baseline/"production_fault_properties.mat")},{quote(out/"synthetic_raw_export_test")});')
    command(out/'bridge_process',[sys.executable,'-B',str(ROOT/'vendor/pipeline/vendor/kr/build_bridge.py'),
        str(ROOT/'vendor/pipeline/vendor/numeric'),str(out/'bridge')])
    julia=['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes']
    command(out/'pc_tests',julia+[str(ROOT/'vendor/pipeline/vendor/pc/test/runtests.jl')])
    command(out/'plan_tests',julia+[str(ROOT/'test_plan.jl'),str(ROOT/'tests/raw.bin'),str(ROOT/'tests/geometry.bin'),str(out/'plan_test_output')])
    command(out/'pipeline_tests',julia+[str(ROOT/'vendor/pipeline/test_pipeline.jl'),str(ROOT/'tests/raw.bin'),
        str(out/'bridge/libphase_bridge.so'),str(out/'pipeline_test_output')])
    command(out/'python_tests',[sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_*.py'])
    check_hashes(Path('/'),inventory);check_hashes(Path('/'),m['sources'])
    write(out/'initialized.json',dict(source=m['source'],baseline=hashes(baseline),bridge=hashes(out/'bridge'),
        inventory_sha256=sha(out/'external_inventory.json'),inputs_lock_sha256=sha(out/'inputs_lock.json'),passed=True))


def accepted(out,r):
    p=out/'accepted'/f'slot{r["slot"]:03d}.json'
    if not p.exists():return None
    d=read(p);assert d['calculation_key']==r['calculation_key'] and d['numerical_checks_passed']
    check_hashes(Path(d['attempt']),d['outputs']);return d


def parse_summary(path):
    # Julia's own TOML parser emits a standard JSON summary for Python 3.10.
    return read(path)


def slot_run(out,m,slot):
    r=m['records'][slot-1];assert r['slot']==slot
    if accepted(out,r):print('SKIP verified completed slot',slot);return
    init=read(out/'initialized.json');assert init['passed'] and init['source']==m['source']
    check_hashes(out/'bridge',init['bridge'])
    assert sha(out/'external_inventory.json')==init['inventory_sha256']
    inv=read(out/'external_inventory.json')
    lock=out/'locks'/f'slot{slot:03d}';lock.parent.mkdir(exist_ok=True);lock.mkdir()
    a=out/'attempts'/f'slot{slot:03d}_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    a.mkdir(parents=True);start=time.monotonic()
    try:
        folder=out/'inputs'/f'slot{slot:03d}';selection=folder/'selection.csv'
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        summary=out/'baseline'/f'slot{slot:03d}/summary.json'
        assert sha(summary)==init['baseline'][str(summary.relative_to(out/'baseline'))]
        write(a/'provenance.json',dict(record=r,source=m['source'],node=socket.gethostname(),
            slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},production_promotion=False))
        # Restart at an immutable completed preparation, never an incomplete replay.
        previous=[]
        for p in sorted((out/'attempts').glob(f'slot{slot:03d}_*/preparation.done.json')):
            d=read(p)
            if d['calculation_key']==r['calculation_key']:
                check_hashes(Path(d['root']),d['files']);previous.append(d)
        if previous:
            prepared=Path(previous[-1]['root']);write(a/'reused_preparation.json',previous[-1])
        else:
            prepared=a/'prepared'
            matlab(a/'prepare_process',f'prepare_slot({quote(selection)},{quote(summary)},{quote(prepared)},'
                f'{quote(PHYSICS)},{quote(DATA)},{quote(MRST)});')
            write(a/'preparation.done.json',dict(root=str(prepared),files=hashes(prepared),calculation_key=r['calculation_key']))
        for name in (prepared/'loaded_matlab_files.txt').read_text().splitlines():
            p=Path(name)
            if str(p).startswith((str(PHYSICS),str(MRST))):
                assert name in inv and sha(p)==inv[name],('Loaded external code changed',name)
        command(a/'pipeline_process',['julia','--startup-file=no','--math-mode=ieee',str(ROOT/'plan_and_run.jl'),
            str(prepared/'raw_unplanned.bin'),str(prepared/'geometry_unplanned.bin'),str(out/'bridge/libphase_bridge.so'),str(a/'numerical'),'false'])
        command(a/'summary_process',['julia','--startup-file=no',str(ROOT/'summarize.jl'),str(a/'numerical/coupled/summary.toml'),str(a/'summary.json')])
        s=parse_summary(a/'summary.json');curves=numeric(a/'numerical/coupled/pc_kr_raw.csv',True)
        result=audit(curves);write(a/'export_audit.json',result)
        canonical=float(r['assignment']['perm_kzz'])*9.869233e-16
        logdiff=abs(math.log10(s['intrinsic_permeability_m2']/canonical));assert logdiff<=.005
        base=out/'baseline/curves'/f'slot{slot:03d}'
        for p in [base/'pc.csv',base/'kr.csv']:
            assert sha(p)==init['baseline'][str(p.relative_to(out/'baseline'))]
        write(a/'comparison.json',compare(curves,numeric(base/'pc.csv'),numeric(base/'kr.csv')))
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        for name in (prepared/'loaded_matlab_files.txt').read_text().splitlines():
            if name in inv:assert sha(name)==inv[name],('Loaded external code changed after solve',name)
        assert subprocess.check_output(['git','-C',str(PHYSICS),'rev-parse','HEAD'],text=True).strip()==PHYSICS_COMMIT
        assert verify()==m['source']
        d=dict(slot=slot,source=m['source'],calculation_key=r['calculation_key'],attempt=str(a),
            numerical_checks_passed=True,canonical_kzz_log10_difference=logdiff,
            export_flags=result['flags'],reservoir_export_approved=False,production_promotion=False,
            elapsed_seconds=time.monotonic()-start,outputs=hashes(a))
        write(a/'completion.json',d);write(out/'accepted'/f'slot{slot:03d}.json',d)
        print('COMPLETE slot',slot,'flags',result['flags'],flush=True)
    except Exception as e:
        write(a/'execution_error.json',dict(error=str(e)));raise
    finally:lock.rmdir()


def gate(out,m):
    good=[accepted(out,m['records'][i-1]) for i in m['pilots']];assert all(good)
    bytes_per=max(sum(p.stat().st_size for p in Path(x['attempt']).rglob('*') if p.is_file()) for x in good)
    estimate=bytes_per*len(m['remaining'])*1.25
    assert shutil.disk_usage(out).free>estimate+50*1024**3,'Insufficient measured project-storage headroom'
    write(out/'pilot_gate.json',dict(passed=True,slots=m['pilots'],estimated_remaining_bytes=estimate,
        source=m['source'],reservoir_export_approved=False,flags_not_job_failures=True))


def assemble(out,m):
    all_done=[accepted(out,r) for r in m['records']];assert all(all_done)
    records=[]
    for r,d in zip(m['records'],all_done):
        x=r['assignment'];a=Path(d['attempt'])
        records.append(dict(slot=r['slot'],window=int(x['window'][4:]),slice=int(x['slice_index']),
            sample=int(x['selected_sample_index']),seed=int(x['exact_replay_seed']),
            curve=str(a/'numerical/coupled/pc_kr_raw.csv'),audit=str(a/'export_audit.json'),comparison=str(a/'comparison.json')))
    write(out/'assembly_manifest.json',dict(source=m['source'],records=records,baseline_mat=str(out/'baseline/production_fault_properties.mat')))
    matlab(out/'assembly_process',f'assemble_raw({quote(out/"assembly_manifest.json")},{quote(out/"raw_export")});')
    check_hashes(Path('/'),m['sources'])
    write(out/'study_complete.json',dict(numerical_assignments=522,full_coverage=True,
        raw_mat_roundtrip=read(out/'raw_export/roundtrip.json'),source=m['source'],
        flagged_assignments=sum(bool(x['export_flags']) for x in all_done),
        production_promotion=False,reservoir_export_approved=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','verify','prepare','initialize','slot','gate','assemble'])
    p.add_argument('output',nargs='?',type=Path);p.add_argument('slot',nargs='?',type=int);a=p.parse_args()
    if a.mode=='seal':f=source_files();write(ROOT/'source_lock.json',dict(files=f,sha256=digest(f)));return
    if a.mode=='verify':print(verify());return
    if a.mode=='prepare':prepare(a.output);return
    assert os.environ.get('SLURM_JOB_ID'),'Compute-node execution required'
    assert subprocess.check_output(['julia','--startup-file=no','--version'],text=True).strip()=='julia version 1.12.6'
    out=scope(a.output);source=verify();check_hashes(out/'inputs',read(out/'inputs_lock.json'))
    m=read(out/'inputs/manifest.json');assert m['source']==source
    if a.mode=='initialize':initialize(out,m)
    elif a.mode=='slot':assert a.slot and 1<=a.slot<=522;slot_run(out,m,a.slot)
    elif a.mode=='gate':gate(out,m)
    else:assemble(out,m)


if __name__=='__main__':main()
