"""Isolated integrated Julia Pc/Kr qualification; never changes production."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
PC_HASH='44276ba3e86731e759110d9d9b570a5bb51389b53ff90c272676a0f320b7bd0d'
KR_HASH='6aff022ff309e6ef8053293c6c9d047edd8e51bbaa430b58270c1669943f7386'
OLD_PC=BASE/'julia_pc_module_20260909_v1_loadfix1'
CORE3_PC=BASE/'julia_pc_module_20260909_v1_eventaudit1'
SOLVER_OUT=BASE/'candidate_solver_benchmark_20260909_v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False)
def identity(files):return hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def files():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p!=ROOT/'source_lock.json'}


def verify():
    lock=read(ROOT/'source_lock.json');found=files()
    assert found==lock['files'] and identity(found)==lock['sha256'],'Integrated source changed'
    for folder,digest in [('pc',PC_HASH),('kr',KR_HASH)]:
        child=ROOT/'vendor'/folder;expected=read(child/'source_lock.json')
        assert expected['sha256']==digest and identity(expected['files'])==digest
        for name,value in expected['files'].items():assert sha(child/name)==value,'Validated component changed'
    return lock['sha256']


def pc_study():
    spec=importlib.util.spec_from_file_location('frozen_pc_study',ROOT/'vendor/pc/validation/study.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def one(root,pattern):
    paths=list(root.glob(pattern));assert len(paths)==1,(root,pattern,paths);return paths[0]


def prepare(out):
    assert BASE in out.resolve().parents and not out.exists()
    source=verify();previous=read(SOLVER_OUT/'inputs/manifest.json');records=[]
    for core in [7,3,18,22,25,47]:
        pc=one(OLD_PC/'attempts',f'core{core:03d}_*')
        qualification=one(CORE3_PC/'attempts','core003_*/completion.json') if core==3 else pc/'completion.json'
        assert read(qualification)['all_equivalence_gates_passed']
        flow=one(SOLVER_OUT/'attempts',f'core{core:03d}_*/completion.json')
        result=read(flow);assert result['matched_accuracy_passed']
        packet=flow.parent/'preparation/phase_inputs.mat'
        exports=read(pc/'input_receipt.json')
        prep=pc/'preparation';reference=Path((prep/'reference_path.txt').read_text().strip())
        if reference==prep/'reference.bin':reference_hash=exports['reference.bin']
        else:
            assert core==25
            reference_hash=read(reference.parent/'completion.json')['outputs'][reference.name]
        item=next(x for x in previous['cores'] if x['core']==core)
        accuracy=Path(item['plan_file']).with_name(f'core{core:03d}_accuracy.json')
        records.append(dict(core=core,item=item,pc_preparation=str(prep),exports=exports,
            reference_file=str(reference),reference_sha256=reference_hash,phase_packet=str(packet),
            phase_packet_sha256=result['outputs']['preparation/phase_inputs.mat'],
            protected_receipts={str(qualification):sha(qualification),str(flow):sha(flow),
                                str(pc/'input_receipt.json'):sha(pc/'input_receipt.json')},
            accuracy_file=str(accuracy),accuracy_sha256=sha(accuracy)))
    controls=one(BASE/'julia_kr_benchmark_20260909_v1_exportfix1/attempts','core007_*/preparation/precision_controls.bin')
    smoke=one(OLD_PC/'attempts','core007_*/matlab_controls/control05/raw.bin')
    (out/'inputs').mkdir(parents=True,exist_ok=False)
    write(out/'inputs/manifest.json',dict(source=source,cores=records,
        precision_controls=str(controls),precision_controls_sha256=sha(controls),
        smoke_raw=str(smoke),smoke_raw_sha256=sha(smoke),production_promotion=False,
        numerical_workflow='Julia Pc/MIP to Julia corrected-direct Kr; MATLAB only packages saved geometry'))


def command(folder,name,args):
    folder.mkdir(parents=True,exist_ok=False);print('STAGE',name,flush=True);start=time.monotonic()
    with (folder/(name+'.log')).open('x') as f:
        p=subprocess.run(['/usr/bin/time','-v','-o',str(folder/(name+'_resources.txt'))]+args,
            cwd=folder,stdout=f,stderr=subprocess.STDOUT,timeout=21000)
    write(folder/(name+'_receipt.json'),dict(returncode=p.returncode,seconds=time.monotonic()-start,args=args))
    assert p.returncode==0,name+' failed; original outputs retained'


def run(out,core):
    assert os.environ.get('SLURM_JOB_ID'),'Compute-node execution only'
    assert BASE in out.resolve().parents
    source=verify();manifest=out/'inputs/manifest.json';mh=sha(manifest);m=read(manifest)
    assert m['source']==source;entry=next(x for x in m['cores'] if x['core']==core)
    attempt=out/'attempts'/f'core{core:03d}_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    attempt.mkdir(parents=True,exist_ok=False);print('ATTEMPT',attempt,flush=True)
    prep=Path(entry['pc_preparation']);raw=prep/'raw.bin'
    def protect():
        assert verify()==source and sha(manifest)==mh
        for p,h in entry['protected_receipts'].items():assert sha(p)==h
        for name,h in entry['exports'].items():assert sha(prep/name)==h
        assert sha(entry['reference_file'])==entry['reference_sha256']
        assert sha(entry['phase_packet'])==entry['phase_packet_sha256']
        assert sha(m['precision_controls'])==m['precision_controls_sha256']
    try:
        protect();old=pc_study().frozen();plan,deps=old.upstream(entry['item'])
        numeric=Path(plan['record']['spec']['numeric_package'])
        for name in ['phase_precision.cpp','phase_precision_repair.cpp']:
            assert sha(ROOT/'vendor/numeric/vendor'/name)==sha(numeric/'vendor'/name)
        assert subprocess.check_output(['julia','--startup-file=no','--version'],text=True).strip()=='julia version 1.12.6'
        write(attempt/'provenance.json',dict(source=source,manifest_sha256=mh,entry=entry,
            dependencies=deps,node=socket.gethostname(),slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},
            production_promotion=False))
        command(attempt/'build','bridge',[sys.executable,'-B',str(ROOT/'vendor/kr/build_bridge.py'),
                str(ROOT/'vendor/numeric'),str(attempt/'bridge')])
        lib=attempt/'bridge/libphase_bridge.so'
        julia=['julia','--startup-file=no','--math-mode=ieee']
        command(attempt/'flow_controls','julia',julia+['--check-bounds=yes',str(ROOT/'vendor/kr/test_flow.jl'),str(lib),m['precision_controls']])
        command(attempt/'pc_controls','julia',julia+['--check-bounds=yes',str(ROOT/'vendor/pc/test/runtests.jl')])
        if core==7:
            assert sha(m['smoke_raw'])==m['smoke_raw_sha256']
            command(attempt/'pipeline_controls','julia',julia+['--check-bounds=yes',str(ROOT/'test_pipeline.jl'),
                m['smoke_raw'],str(lib),str(attempt/'synthetic_pipeline')])
        quote=old.quote;geometry=attempt/'inputs'
        expr=f'addpath({quote(ROOT)});export_geometry({quote(entry["phase_packet"])},{quote(sha(raw))},{quote(geometry)});'
        command(attempt/'geometry_export','matlab',['matlab','-batch',expr])
        exports={p.name:sha(p) for p in geometry.iterdir()}
        write(attempt/'input_receipt.json',exports)
        command(attempt/'pipeline_process','julia',julia+[str(ROOT/'run_core.jl'),str(raw),str(geometry/'geometry.bin'),
            str(lib),str(attempt/'coupled'),'--states'])
        command(attempt/'validation_process','julia',julia+[str(ROOT/'validate_pipeline.jl'),str(raw),str(geometry),
            str(prep),str(attempt/'coupled'),str(attempt/'comparison.json')])
        result=read(attempt/'comparison.json');assert result['all_coupled_checks_passed']
        accuracy=Path(entry['accuracy_file']);assert sha(accuracy)==entry['accuracy_sha256']
        write(attempt/'sampling_accuracy.json',read(accuracy))
        _,after=old.upstream(entry['item']);assert after['sha256']==deps['sha256'];protect()
        for name,h in exports.items():assert sha(geometry/name)==h
        write(attempt/'completion.json',dict(core=core,source=source,execution_complete=True,
            all_coupled_checks_passed=True,original_inputs_unchanged=True,numerical_components_unchanged=True,
            production_promotion=False,reservoir_export_approved=False,
            outputs={p.relative_to(attempt).as_posix():sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()}))
        print('COMPLETE coupled Julia Pc/Kr core',core,'max relative Kr',result['max_relative_kr_difference'],flush=True)
    except Exception as err:
        write(attempt/'execution_error.json',dict(error=str(err)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    s.add_parser('seal');s.add_parser('verify')
    q=s.add_parser('prepare');q.add_argument('output',type=Path)
    q=s.add_parser('run');q.add_argument('output',type=Path);q.add_argument('core',type=int)
    a=p.parse_args()
    if a.mode=='seal':f=files();write(ROOT/'source_lock.json',dict(files=f,sha256=identity(f)))
    elif a.mode=='verify':print(verify())
    elif a.mode=='prepare':prepare(a.output)
    else:run(a.output,a.core)
