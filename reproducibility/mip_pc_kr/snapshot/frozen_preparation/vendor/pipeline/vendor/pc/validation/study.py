"""Isolated qualification of the standalone Julia Pc module; no phase-flow solves."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
SOLVER=BASE/'releases/solver_benchmark_b9868072c55e'
SOLVER_HASH='b9868072c55e92111503185a08ae53c85d1910acea3702f8b321205045fa6505'
MIP_HASH='5d1e9a456f673e920eddc605f69e126a289b2def3f4daf055e1478f318ca992a'
ENGINE_HASH='b3c35054eb87c0c69b869bc600e85911880efc336a22d8cb64e84e64ebfc94b0'
CORE25=BASE/'julia_mip_core25_20260909_v1_fullretry1/attempts/core025_job22388045_1788967256393362586'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def files():
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name!='source_lock.json'}

def identity(data):return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify():
    lock=read(ROOT/'source_lock.json');found=files()
    assert found==lock['files'] and identity(found)==lock['sha256'],'Qualification source changed'
    assert sha(ROOT/'vendor/StandaloneMIP.jl')==ENGINE_HASH,'Validated MIP engine changed'
    return lock['sha256']

def frozen():
    spec=importlib.util.spec_from_file_location('frozen_solver_study',SOLVER/'study.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert m.verify()==SOLVER_HASH
    return m

def prepare(out):
    assert BASE in out.resolve().parents and not out.exists()
    old=BASE/'candidate_solver_benchmark_20260909_v1/inputs/manifest.json'
    m=read(old);assert m['source']==SOLVER_HASH
    done=read(CORE25/'completion.json');assert done['all_bitwise_equal'] and done['previous_outputs_unchanged']
    reuse={}
    for name in ['compiled_reference.bin','core25_packet.mat']:
        digest=sha(CORE25/name);assert done['outputs'][name]==digest;reuse[name]=digest
    items=[]
    for number in [7,3,18,22,25,47]:
        item=next(x for x in m['cores'] if x['core']==number)
        assert sha(item['plan_file'])==item['plan_sha256']
        accuracy=Path(item['plan_file']).with_name(f'core{number:03d}_accuracy.json')
        items.append(dict(core=number,item=item,accuracy_file=str(accuracy),accuracy_sha256=sha(accuracy)))
    (out/'inputs').mkdir(parents=True,exist_ok=False)
    write(out/'inputs/manifest.json',dict(source=verify(),cores=items,solver_source=SOLVER_HASH,
        previous_manifest=str(old),previous_manifest_sha256=sha(old),core25_reuse=str(CORE25),
        core25_completion_sha256=sha(CORE25/'completion.json'),core25_files=reuse,
        pc_only=True,flow_solves=False,reservoir_simulations=False,production_promotion=False))

def command(folder,name,args):
    folder.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    print('STAGE',name,flush=True)
    with (folder/(name+'.log')).open('x') as log:
        proc=subprocess.run(['/usr/bin/time','-v','-o',str(folder/(name+'_resources.txt'))]+args,
            cwd=folder,stdout=log,stderr=subprocess.STDOUT,timeout=17500)
    write(folder/(name+'_receipt.json'),dict(returncode=proc.returncode,seconds=time.monotonic()-start,args=args))
    assert proc.returncode==0,name+' failed; original logs retained'

def check_reuse(manifest):
    root=Path(manifest['core25_reuse'])
    assert sha(root/'completion.json')==manifest['core25_completion_sha256']
    for name,digest in manifest['core25_files'].items():assert sha(root/name)==digest

def core(output,number):
    assert os.environ.get('SLURM_JOB_ID'),'Compute-node execution only'
    assert BASE in output.resolve().parents
    source=verify();manifest=read(output/'inputs/manifest.json');mh=sha(output/'inputs/manifest.json')
    assert manifest['source']==source and not manifest['flow_solves']
    entry=next(x for x in manifest['cores'] if x['core']==number);item=entry['item']
    out=output/'attempts'/f'core{number:03d}_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=False);print('ATTEMPT',out,'VERIFYING INPUTS',flush=True)
    try:
        old=frozen();plan,deps=old.upstream(item);spec=plan['record']['spec'];numeric=spec['numeric_package']
        assert subprocess.check_output(['julia','--startup-file=no','--version'],text=True).strip()=='julia version 1.12.6'
        if number==25:check_reuse(manifest)
        write(out/'provenance.json',dict(core=number,source=source,manifest_sha256=mh,inputs=entry,
            node=socket.gethostname(),cpu_info=Path('/proc/cpuinfo').read_text(),dependencies=deps,
            slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},production_promotion=False))
        quote=old.quote;validation=ROOT/'validation'
        command(out/'package_load','julia',['julia','--startup-file=no','--project='+str(ROOT),'-e','using FaultCorePc; println(:PACKAGE_LOAD_OK)'])
        command(out/'unit_tests','julia',['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',str(ROOT/'test/runtests.jl')])
        if number==7:
            expr=f'addpath({quote(validation)});export_controls({quote(numeric)},{quote(out/"matlab_controls")});'
            command(out/'controls_export','matlab',['matlab','-batch',expr])
            command(out/'controls_process','julia',['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',
                str(validation/'validate.jl'),'controls',str(out/'matlab_controls'),str(out/'julia_controls')])
            assert read(out/'julia_controls/controls.json')['all_passed']
        prep=out/'preparation';reuse_ref='';reuse_model=''
        if number==25:
            reuse_ref=str(CORE25/'compiled_reference.bin');reuse_model=str(CORE25/'core25_packet.mat')
        expr=(f'addpath({quote(validation)});export_core({quote(item["plan_file"])},'
              f'{quote(item["accepted_attempt"])},{quote(prep)},{quote(reuse_ref)},{quote(reuse_model)});')
        command(prep,'matlab',['matlab','-batch',expr])
        exports={p.name:sha(p) for p in prep.iterdir() if p.is_file()}
        write(out/'input_receipt.json',exports)
        command(out/'julia_process','julia',['julia','--startup-file=no','--math-mode=ieee',str(validation/'validate.jl'),
            'core',str(prep),str(out/'julia')])
        result=read(out/'julia/comparison.json');assert result['all_equivalence_gates_passed'] and result['points']==100
        expr=(f'addpath({quote(validation)});check_archive({quote(numeric)},{quote(prep/"model.bin")},'
            f'{quote(out/"julia")},{quote(exports["raw.bin"])},{quote(out/"archive_check/pass.json")});')
        command(out/'archive_check','matlab',['matlab','-batch',expr])
        assert read(out/'archive_check/pass.json')['streaming_matlab_consumer_passed']
        accuracy=Path(entry['accuracy_file']);assert sha(accuracy)==entry['accuracy_sha256']
        write(out/'sampling_accuracy.json',read(accuracy))
        _,after=old.upstream(item);assert after['sha256']==deps['sha256']
        if number==25:check_reuse(manifest)
        assert verify()==source and sha(output/'inputs/manifest.json')==mh
        for name,digest in exports.items():assert sha(prep/name)==digest
        write(out/'completion.json',dict(core=number,source=source,execution_complete=True,
            all_equivalence_gates_passed=True,native_archive_validated_in_julia_and_matlab=True,
            reference_reused=number==25,production_promotion=False,reservoir_export_approved=False,
            outputs={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
        print('COMPLETE',number,'max Sg',result['max_absolute_sg_difference'],'exact topology; no production changes',flush=True)
    except Exception as err:
        write(out/'execution_error.json',dict(error=str(err)));raise

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='mode',required=True)
    s.add_parser('seal');s.add_parser('verify')
    q=s.add_parser('prepare');q.add_argument('output',type=Path)
    q=s.add_parser('core');q.add_argument('output',type=Path);q.add_argument('number',type=int)
    a=p.parse_args()
    if a.mode=='seal':
        f=files();write(ROOT/'source_lock.json',dict(files=f,sha256=identity(f)))
    elif a.mode=='verify':print(verify())
    elif a.mode=='prepare':prepare(a.output)
    else:core(a.output,a.number)

if __name__=='__main__':main()
