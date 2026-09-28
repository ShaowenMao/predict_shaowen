"""Paired standalone Julia versus corrected MATLAB direct phase-flow study."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
SOLVER=BASE/'releases/solver_benchmark_b9868072c55e'
SOLVER_HASH='b9868072c55e92111503185a08ae53c85d1910acea3702f8b321205045fa6505'


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


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify():
    lock=read(ROOT/'source_lock.json');found=files()
    assert lock['files']==found and lock['sha256']==identity(found),'Benchmark source changed'
    return lock['sha256']


def frozen():
    s=importlib.util.spec_from_file_location('sealed_solver_benchmark',SOLVER/'study.py')
    p=importlib.util.module_from_spec(s);s.loader.exec_module(p);assert p.verify()==SOLVER_HASH
    return p


def prepare(previous,output):
    p=frozen();old=read(previous/'inputs/manifest.json');assert old['source']==SOLVER_HASH
    cores=[]
    for number in [7,18,3]:
        items=[x for x in old['cores'] if x['core']==number];assert len(items)==1
        attempts=[a for a in (previous/'attempts').glob(f'core{number:03d}_*') if (a/'completion.json').exists()]
        assert len(attempts)==1;a=attempts[0];done=read(a/'completion.json')
        assert done['source']==SOLVER_HASH and done['matched_accuracy_passed']
        packet=a/'preparation/phase_inputs.mat';digest=sha(packet)
        assert done['outputs']['preparation/phase_inputs.mat']==digest==read(a/'packet_receipt.json')['sha256']
        rows=[json.loads(s) for s in (a/'direct_rep1/calls.jsonl').read_text().splitlines()]
        candidates=[x for x in rows if x['phase']!='absolute'];selected=max(candidates,key=lambda x:x['seconds'])
        cores.append(dict(core=number,item=items[0],previous_attempt=str(a),
            previous_completion_sha256=sha(a/'completion.json'),packet=str(packet),packet_sha256=digest,
            profile_sample=selected['sample'],profile_phase=selected['phase']))
    (output/'inputs').mkdir(parents=True,exist_ok=False)
    write(output/'inputs/manifest.json',dict(source=verify(),cores=cores,repetitions=3,
          previous_manifest=str(previous/'inputs/manifest.json'),previous_manifest_sha256=sha(previous/'inputs/manifest.json'),
          solver_source=SOLVER_HASH,production_promotion=False))


def command(folder,name,args):
    folder.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    with (folder/(name+'.log')).open('x') as f:
        proc=subprocess.run(['/usr/bin/time','-v','-o',str(folder/(name+'_resources.txt'))]+args,
                            cwd=folder,stdout=f,stderr=subprocess.STDOUT,timeout=17500)
    write(folder/(name+'_receipt.json'),dict(returncode=proc.returncode,seconds=time.monotonic()-began,args=args))
    assert proc.returncode==0,name+' failed; original logs retained'


def rel(a,b):return abs(a-b)/max(abs(a),abs(b),sys.float_info.min)


def rows(folder):
    data=[json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines() if line.strip()]
    keys=[(x['sample'],x['phase']) for x in data];assert len(set(keys))==len(keys),'Duplicate calls'
    return dict(zip(keys,data))


def compare_pair(a,b):
    assert a.keys()==b.keys() and len(a)>0,'Mismatched call sets'
    differences=[]
    for key in a:
        x,y=a[key],b[key]
        assert x['accepted'] and y['accepted'] and x['reference_matches'] and y['reference_matches']
        assert math.isfinite(x['q']) and math.isfinite(y['q']) and min(x['q'],y['q'])>=0
        difference=rel(x['q'],y['q'])
        assert (x['q']==0)==(y['q']==0) and difference<=1e-5,('Corrected conductance mismatch',key,difference)
        differences.append(dict(sample=key[0],phase=key[1],relative_difference=difference))
    return differences


def summarize(out):
    mapping=read(out/'preparation/mapping.json');runs=[];pairs=[];curve_max=0.0
    samples=mapping['unique_sample'];expected={(0,'absolute')}|{(s,p) for s in samples for p in ['brine','gas']}
    for rep in range(1,4):
        by={}
        for backend in ['matlab','julia']:
            folder=out/f'{backend}_rep{rep}';data=rows(folder);summary=read(folder/'summary.json')
            assert summary['all_calls_accepted'] and summary['fixed100_points']==100 and data.keys()==expected
            seconds=sum(v['seconds'] for k,v in data.items() if k[1]!='absolute')
            assert abs(seconds-summary['phase_seconds'])<1e-6
            by[backend]=data
            runs.append(dict(backend=backend,repeat=rep,phase_seconds=seconds,
                absolute_seconds=data[0,'absolute']['seconds'],
                fallback_calls=sum(bool(x['info'].get('fallback_used',False)) for k,x in data.items() if k[1]!='absolute')))
        differences=compare_pair(by['matlab'],by['julia']);pairs.extend(dict(repeat=rep,**x) for x in differences)
        curves=read(out/f'julia_rep{rep}/curves.json')['curves'];assert len(curves)==100
        for point,owner in enumerate(mapping['owner']):
            sample=samples[owner-1]
            for phase,label in [('brine','krw'),('gas','krg')]:
                ref=by['matlab'][sample,phase]['q']/mapping['qa_ref']
                observed=curves[point][label]
                assert rel(ref,observed)<=1e-5 and (ref==0)==(observed==0)
                curve_max=max(curve_max,abs(ref-observed))
    med={b:statistics.median(r['phase_seconds'] for r in runs if r['backend']==b) for b in ['matlab','julia']}
    assert read(out/'assembly_check/assembly_checks.json')['all_parts_bitwise_equal']
    return dict(core=mapping['core'],runs=runs,phase_time_medians=med,
        julia_speedup=med['matlab']/med['julia'],pairs=pairs,
        maximum_relative_conductance_difference=max(x['relative_difference'] for x in pairs),
        maximum_absolute_kr_difference=curve_max,matched_accuracy_passed=True,
        full_curve_points=100,repetitions=3,production_promotion=False)


def check_packet(entry):
    old=Path(entry['previous_attempt']);assert sha(old/'completion.json')==entry['previous_completion_sha256']
    assert sha(entry['packet'])==entry['packet_sha256']
    done=read(old/'completion.json')
    assert done['outputs']['preparation/phase_inputs.mat']==entry['packet_sha256'] and done['matched_accuracy_passed']


def core(output,number):
    assert os.environ.get('SLURM_JOB_ID'),'Compute-node execution only'
    output=output.resolve();assert BASE in output.parents
    source=verify();m=read(output/'inputs/manifest.json');manifest_sha=sha(output/'inputs/manifest.json')
    assert m['source']==source and m['repetitions']==3
    entries=[x for x in m['cores'] if x['core']==number];assert len(entries)==1;entry=entries[0]
    out=output/'attempts'/f'core{number:03d}_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=False)
    print('ATTEMPT',out,'VERIFYING INPUTS',flush=True);start=time.monotonic()
    try:
        p=frozen();check_packet(entry);plan,deps=p.upstream(entry['item']);spec=plan['record']['spec']
        assert subprocess.check_output(['julia','--startup-file=no','--version'],text=True).strip()=='julia version 1.12.6'
        write(out/'provenance.json',dict(core=number,source=source,manifest_sha256=manifest_sha,input=entry,
            dependencies=deps,node=socket.gethostname(),cpu_info=Path('/proc/cpuinfo').read_text(),
            slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')},production_promotion=False))
        command(out,'build_bridge',['python','-B',str(ROOT/'build_bridge.py'),spec['numeric_package'],str(out/'bridge')])
        lib=out/'bridge/libphase_bridge.so'
        prep=out/'preparation'
        expression=(f'addpath({p.quote(ROOT)});export_packet({p.quote(spec["numeric_package"])},'
            f'{p.quote(entry["packet"])},{p.quote(prep)},{entry["profile_sample"]},{p.quote(entry["profile_phase"])});')
        command(prep,'export',['matlab','-batch',expression])
        exports={path.name:sha(path) for path in prep.glob('*.bin')}
        write(out/'packet_receipt.json',exports)
        command(out/'controls','julia',['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',
            str(ROOT/'test_flow.jl'),str(lib),str(prep/'precision_controls.bin')])
        command(out/'assembly_check','julia',['julia','--startup-file=no','--math-mode=ieee',str(ROOT/'run_julia.jl'),
            str(prep),str(lib),str(out/'assembly_check'),'check'])
        for rep in range(1,4):
            order=['matlab','julia'] if (number+rep)%2==0 else ['julia','matlab']
            for backend in order:
                folder=out/f'{backend}_rep{rep}'
                if backend=='matlab':
                    expr=(f'addpath({p.quote(SOLVER)});solver_worker({p.quote(entry["item"]["plan_file"])},'
                        f'{p.quote(entry["packet"])},{p.quote(folder)},\'direct\',{rep});')
                    command(folder,'worker',['matlab','-batch',expr])
                else:
                    command(folder,'worker',['julia','--startup-file=no','--math-mode=ieee',str(ROOT/'run_julia.jl'),
                        str(prep),str(lib),str(folder),'run',str(rep)])
                assert read(folder/'summary.json')['all_calls_accepted']
                print('WORKER COMPLETE',number,backend,rep,flush=True)
            compare_pair(rows(out/f'matlab_rep{rep}'),rows(out/f'julia_rep{rep}'))
        result=summarize(out);write(out/'comparison.json',result)
        check_packet(entry);_,after=p.upstream(entry['item']);assert deps['sha256']==after['sha256']
        for name,digest in exports.items():assert sha(prep/name)==digest
        assert verify()==source and p.verify()==SOLVER_HASH and sha(output/'inputs/manifest.json')==manifest_sha
        write(out/'completion.json',dict(core=number,execution_complete=True,matched_accuracy_passed=True,source=source,
              wrapper_seconds=time.monotonic()-start,outputs={x.relative_to(out).as_posix():sha(x) for x in sorted(out.rglob('*')) if x.is_file()},
              production_promotion=False))
        print('COMPLETE',number,'speedup',result['julia_speedup'],'max relative Q error',result['maximum_relative_conductance_difference'],flush=True)
    except Exception as e:
        write(out/'execution_error.json',dict(error=str(e)));raise


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    sub.add_parser('seal');sub.add_parser('verify')
    q=sub.add_parser('prepare');q.add_argument('previous',type=Path);q.add_argument('output',type=Path)
    q=sub.add_parser('core');q.add_argument('output',type=Path);q.add_argument('number',type=int)
    a=parser.parse_args()
    if a.mode=='seal':
        x=files();write(ROOT/'source_lock.json',dict(files=x,sha256=identity(x)))
    elif a.mode=='verify':print(verify())
    elif a.mode=='prepare':prepare(a.previous,a.output)
    else:core(a.output,a.number)


if __name__=='__main__':main()
