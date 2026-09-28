"""Bounded isolated qualification only. No campaign ledger or production writes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from resource import cgroup, parse_allocation, passed, snapshot

HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''): h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    content=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    p=Path(p)
    if p.exists():
        assert p.read_text()==content,'Refuse overwrite: '+str(p)
    else:
        with p.open('x') as f:
            f.write(content); f.flush(); os.fsync(f.fileno())


def pins_ok(pins):
    for p,h in pins.items():
        assert sha(p)==h, 'Changed source/input: '+p


def spec():
    m=read(HERE/'manifest.json')
    assert str(HERE)==m['remote'] and m['qualification_only'] and not m['campaign_resume']
    pins_ok({str(HERE/n):h for n,h in m['sources'].items()})
    return m


def prepare():
    m=spec(); pins=dict(m['pins'])
    pins.update({str(HERE/n):h for n,h in m['sources'].items()})
    entries={}
    for key,c in m['cases'].items():
        pins.update(c['pins'])
        audit=tomllib.loads(Path(c['scenario_audit']).read_text())
        assert audit['passed'] and audit['core']==c['slot']
        assert audit['entry_pressure_pa']>0 and audit['clay_swi']==.2
        pins[c['scenario_audit']]=sha(c['scenario_audit'])
        entries[key]=audit['entry_pressure_pa']
    origin=tomllib.loads((HERE/'origin.toml').read_text())
    pins.update(origin['input_hashes'])
    pins_ok(pins)
    for folder in ('logs','stage_receipts','configs','runs','cache','tmp','intents'):
        (HERE/folder).mkdir(exist_ok=True)
    write(HERE/'PREPARED.json',dict(passed=True,manifest_sha256=sha(HERE/'manifest.json'),
                                  pins=pins,entries=entries))
    print('SOURCE_PREPARATION_PASS',len(pins),flush=True)


def stage_policy(m, stage):
    if stage=='bootstrap':return dict(cpus=2,memory_mib=4608,minutes=30)
    if stage=='collect':return dict(cpus=1,memory_mib=768,minutes=30)
    if stage=='native':return dict(cpus=1,memory_mib=3072,minutes=30)
    return m['batches'][stage]


def stage_dependencies(stage, jobs):
    if stage=='bootstrap':return []
    if stage in ('small','larger','prefix'):return [jobs['bootstrap']['job']]
    if stage=='resume':return [jobs['prefix']['job']]
    if stage=='collect':return [jobs[x]['job'] for x in ('small','larger','resume')]
    if stage=='native':return [jobs['collect']['job']]
    raise ValueError(stage)


def submit():
    m=spec(); q=read(HERE/'PREPARED.json')
    assert q['manifest_sha256']==sha(HERE/'manifest.json');pins_ok(q['pins'])
    # An intent left without a receipt is an ambiguous submission, never a retry.
    jobs={}
    for stage in ('bootstrap','small','larger','prefix','resume','collect','native'):
        receipt=HERE/'intents'/(stage+'.submitted.json')
        if receipt.exists():
            jobs[stage]=read(receipt);continue
        intent=HERE/'intents'/(stage+'.intent.json')
        assert not intent.exists(),'Reconcile ambiguous sbatch before resubmitting '+stage
        p=stage_policy(m,stage); deps=stage_dependencies(stage,jobs)
        args=['sbatch','--parsable','--partition=mit_normal','--account=mit_amf_advanced_cpu',
              '--qos=mit_amf_advanced_cpu','--nodes=1','--ntasks=1',
              '--cpus-per-task='+str(p['cpus']),'--mem='+str(p['memory_mib'])+'M',
              '--time='+str(p['minutes']),'--exclude=node1622',
              '--job-name=kr10q_'+stage,'--output='+str(HERE/'logs'/(stage+'_%j.out'))]
        if deps:args+=['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes']
        args += [str(HERE/'job.sh'),str(HERE),stage]
        write(intent,dict(command=args,manifest_sha256=q['manifest_sha256']))
        r=subprocess.run(args,text=True,capture_output=True,check=True,timeout=60)
        job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
        # Persist the returned ID before any verification that could fail.
        write(HERE/'intents'/(stage+'.returned.json'),dict(job=job,stdout=r.stdout))
        actual=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
        parse_allocation(actual,p['cpus'],p['memory_mib'])
        item=dict(stage=stage,job=job,command=args,cpus=p['cpus'],memory_mib=p['memory_mib'],
                  dependencies=deps,actual_scontrol=actual)
        write(receipt,item);jobs[stage]=item
        print(stage,job,p['cpus'],p['memory_mib'],flush=True)
    write(HERE/'SUBMITTED.json',jobs)


def config(m,q,stage,p,out):
    d={k:m[k] for k in ('pipeline','pipeline_hash','bridge','backend')}
    d.update(mode='bootstrap' if stage=='bootstrap' else 'batch',output=str(out),
             memory_mib=p['memory_mib'],source_lock_sha256=q['manifest_sha256'],
             stop_after=p.get('stop_after',0))
    text=''.join(k+' = '+json.dumps(v)+'\n' for k,v in d.items())
    for i,key in enumerate(p.get('keys',[]),1):
        c=m['cases'][key]
        row={k:c[k] for k in ('key','raw','geometry','reference_pc','reference_kr','scenario_audit')}
        if i==1 and stage in m.get('recoveries',{}):
            old=m['recoveries'][stage]
            assert old['key']==key
            row['recover_artifacts']=old['folder']
        if p.get('reuse_prefix') and i<=2:
            old=read(HERE/'stage_receipts/prefix.json')
            assert old['passed']
            path=Path(old['output'])/(f'{i:02d}_'+key)/'CORE_PASS.toml'
            row['reuse_receipt']=str(path)
        text+='\n[[cases]]\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in row.items())
    text+='\n[pins]\n'+''.join(json.dumps(k)+' = '+json.dumps(v)+'\n' for k,v in q['pins'].items())
    path=HERE/'configs'/(stage+'_'+os.environ['SLURM_JOB_ID']+'.toml')
    with path.open('x') as f:f.write(text)
    return path


def validate_core(path):
    d=tomllib.loads(Path(path).read_text());assert d['passed']
    for n,h in d['artifacts'].items():assert sha(Path(path).parent/n)==h
    return d


def run(stage):
    m=spec();q=read(HERE/'PREPARED.json')
    assert q['manifest_sha256']==sha(HERE/'manifest.json');pins_ok(q['pins'])
    p=stage_policy(m,stage);job=os.environ['SLURM_JOB_ID']
    actual=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
    parse_allocation(actual,p['cpus'],p['memory_mib'])
    group=cgroup(job);out=HERE/'runs'/(stage+'_job'+job)
    assert not out.exists(); started=time.monotonic()
    env=dict(os.environ)
    env.update(JULIA_NUM_THREADS=str(p['cpus']),OPENBLAS_NUM_THREADS=str(p['cpus']),
               OMP_NUM_THREADS=str(p['cpus']),MKL_NUM_THREADS=str(p['cpus']))
    if stage in ('bootstrap','small','larger','prefix','resume'):
        own=HERE/'cache/depot' if stage=='bootstrap' else HERE/'job_depots'/job
        own.mkdir(parents=True,exist_ok=True)
        env['JULIA_DEPOT_PATH']=':'.join(dict.fromkeys(map(str,[own,HERE/'cache/depot',m['depot']])))
        cfg=config(m,q,stage,p,out)
        base=[m['julia'],'--startup-file=no','--math-mode=ieee','--check-bounds=yes',
              '--threads=2','--gcthreads=1','--cpu-target=generic','--project='+m['project']]
        args=base+([] if stage=='bootstrap' else ['--compiled-modules=strict','--pkgimages=existing'])
        subprocess.run(args+[str(HERE/'worker.jl'),str(cfg)],env=env,check=True,timeout=(p['minutes']*60-120))
        if stage=='bootstrap':
            subprocess.run(base+['--compiled-modules=strict','--pkgimages=existing',str(HERE/'worker.jl'),str(cfg)],
                           env=env,check=True,timeout=600)
        else:
            mark=out/('PREFIX_PASS.toml' if stage=='prefix' else 'BATCH_PASS.toml')
            d=tomllib.loads(mark.read_text());assert d['passed']
            for r in d['records']:
                assert sha(r['receipt'])==r['receipt_sha256'];validate_core(r['receipt'])
    elif stage=='collect':
        out.mkdir(); batch_records=[]
        for name in ('small','larger','resume'):
            r=read(HERE/'stage_receipts'/(name+'.json'));assert r['passed']
            d=tomllib.loads((Path(r['output'])/'BATCH_PASS.toml').read_text())
            for item in d['records']:
                assert sha(item['receipt'])==item['receipt_sha256'];validate_core(item['receipt'])
                batch_records.append(dict(item,batch=name))
        packet=dict(entries=q['entries'],records=batch_records,manifest_sha256=q['manifest_sha256'])
        write(out/'inputs.json',packet)
        subprocess.run([m['exporter_python'],'-B',str(HERE/'collect.py'),str(out)],env=env,check=True,
                       timeout=p['minutes']*60-120)
    elif stage=='native':
        collect=read(HERE/'stage_receipts/collect.json');assert collect['passed']
        review=read(Path(collect['output'])/'EXPORT_PASS.json');assert review['passed']
        tables=[r['table_path'] for r in review['cores']]
        pins_ok({r['table_path']:r['table_sha256'] for r in review['cores']})
        own=HERE/'native_depots'/job;own.mkdir(parents=True)
        env['JULIA_DEPOT_PATH']=str(own)+':/home/shaowen/.julia'
        args=[m['julia'],'--startup-file=no','--threads=1','--gcthreads=1',
              '--project='+m['native_runtime'],'--compiled-modules=strict','--pkgimages=existing',
              str(HERE/'native.jl'),str(out),*tables]
        subprocess.run(args,env=env,check=True,timeout=p['minutes']*60-120)
        assert tomllib.loads((out/'constitutive_pass.toml').read_text())['passed']
        pins_ok({r['table_path']:r['table_sha256'] for r in review['cores']})
    else:raise ValueError(stage)
    pins_ok(q['pins']); memory=snapshot(group)
    receipt=dict(stage=stage,job=job,output=str(out),elapsed_seconds=time.monotonic()-started,
                 numerical_or_stage_work_complete=True,memory=memory,passed=passed(memory,p['memory_mib']),
                 manifest_sha256=q['manifest_sha256'],actual_scontrol=actual,
                 production_modified=False,campaign_resumed=False)
    write(HERE/'stage_receipts'/(stage+'.json'),receipt)
    print(json.dumps(receipt),flush=True)
    assert receipt['passed'],'Whole-job memory gate; preserve completed work'


if __name__=='__main__':
    if sys.argv[1]=='run':run(sys.argv[2])
    else:{'prepare':prepare,'submit':submit}[sys.argv[1]]()
