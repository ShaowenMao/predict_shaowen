"""Promote individually qualified fixed-scope archives; never run a solver."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tomllib
import unittest
import zipfile
import common as C
import recovery as R
import scope

P=R.P;METHOD=R.METHOD
CONFIG='7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36'
STATES=[(P/'STATE.json',2592),(P/'round2_20261109_v1/solve/STATE.json',125),(P/'round3_20261110_v1/solve/STATE.json',55)]

def stopped():
    for path,cursor in STATES:C.require(C.read(path)==dict(active=[],cursor=cursor),'Campaign resumed; reconcile')

def verify_archive(path,record,q):
    C.require(q['passed'] and q['key']==C.key(record) and q['method']==METHOD,'Qualification identity')
    C.require(q['completed_calls_repeated']==0 and q['production_writes']==0,'Qualification scope changed')
    if q['key'] in scope.NEW_KEYS:
        C.require(q['saved_primary_solves_repeated']==0 and q['new_reference_solves']==1,'Reference-only recovery required')
    C.require(C.sha(path)==q['receipt']['archive_sha256'],'Qualified archive changed')
    C.require(C.archive_receipt(path,record,METHOD)==q['receipt'],'Archive receipt mismatch')
    with zipfile.ZipFile(path) as z:
        rd=lambda n:json.loads(z.read(n))
        td=lambda n:tomllib.loads(z.read(n).decode())
        C.require(rd('input/record.json')==record,'Archived input assignment differs')
        f=td('flow/CORE_PASS.toml');s=td('flow/summary.toml');n=td('NATIVE_PASS.toml');e=rd('export/EXPORT_PASS.json')
        C.require(f['passed'] and f['method']==METHOD and f['record_sha256']==C.digest(record),'Flow receipt identity')
        C.require(f['input']==td('input/INPUT_PASS.toml'),'Flow input mismatch')
        C.require(f['recovery_implementation']==q['source'],'Qualified solver implementation differs')
        for name,h in f['artifacts'].items():C.require(hashlib.sha256(z.read('flow/'+name)).hexdigest()==h,'Flow artifact mismatch')
        C.require(s==q['summary'] and s['mode']=='adaptive10' and s['pc_points']==100 and 2<=s['kr_states']<=10,'Adaptive summary differs')
        C.require(s['completed_flow_calls_repeated']==0 and s['same_mip_microstates'] and s['cached_states_verified']==100,'Reuse/cache proof missing')
        C.require(n['passed'] and e['passed'] and n['record_sha256']==C.digest(record) and n['method']==METHOD,'Final checks absent')
        C.require(hashlib.sha256(z.read('export/sgof.csv')).hexdigest()==e['sha256']==n['sgof_sha256'],'Imported table differs')
        C.require(hashlib.sha256(z.read('NATIVE_PASS.toml')).hexdigest()==q['native_sha256'],'Native receipt changed')
        audit=td('recovery/PRECISION_REAUDIT.toml')
        C.require(audit==q['saved_precision_reaudit'] and audit['passed'] and audit['new_solves']==0,'Saved pressure proof absent')
        C.require(q['saved_precision_solves_repeated']==0,'Passing precision calculation repeated')
    return q['receipt']

def promote(private,target,record,q):
    verify_archive(private,record,q)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():C.require(C.sha(target)==q['receipt']['archive_sha256'],'Existing result conflicts; never overwrite')
    else:os.link(private,target)
    C.require(C.archive_receipt(target,record,METHOD)==q['receipt'],'Published archive verification')
    return q['receipt']

def fingerprint(p):
    C.require(p.is_file() and not p.is_symlink(),'Unsafe saved artifact')
    s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def case_inventory(case,full_hash=True):
    plan=C.read(P/'plans'/(case+'.json'));records=plan['records']
    C.require(len(records)==522 and {r['slot'] for r in records}==set(range(1,523)),'Case coverage')
    missing=[];protected={}
    for r in records:
        C.require(r['case_id']==case and r['configuration_sha256']==CONFIG,'Case configuration')
        a=P/'results'/case/f"core{r['slot']:03d}.zip";d=P/'runs'/case/'done'/f"core{r['slot']:03d}.json"
        if not a.exists() and not d.exists():missing.append(r['slot']);continue
        C.require(a.exists() and d.exists(),'Uncertain existing publication')
        receipt=C.read(d)
        C.require(receipt['key']==C.key(r) and receipt['method']==METHOD and receipt['record_sha256']==C.digest(r) and receipt['scientific_pass'],'Sibling receipt identity')
        C.require(a.stat().st_size==receipt['bytes'],'Sibling archive length')
        if full_hash:C.require(C.archive_receipt(a,r,METHOD)==receipt,'Sibling archive integrity')
        protected[str(a)]=fingerprint(a);protected[str(d)]=fingerprint(d)
    return missing,protected

def reconcile(row):
    case=row['case_id'];jobs=set()
    roots=[(P,'num_'),(P/'round2_20261109_v1/finish','r2f_'),(P/'round2_20261109_v1/solve','r2s_'),(P/'round3_20261110_v1/solve','r3s_')]
    for root,label in roots:
        launch=root/'cases'/case/'LAUNCHED.json'
        if not launch.exists():continue
        d=C.read(launch);t=C.read(root/'cases'/case/'TERMINAL.json')
        C.require(d['job']==t['job'] and all(v[0] in R.TERMINAL for v in t['tasks'].values()),'Nonterminal assignment')
        for suffix,job in [('',d['job']),('_gate',d['gate'])]:
            folder=root/'submissions'/(label+case+suffix)
            ack=C.read(folder/'ACK.json');intent=C.read(folder/'INTENT.json')
            C.require(ack['job']==job and ack['spec']==intent['spec'],'Submission evidence differs')
            C.require(C.read(folder/'RELEASED.json')['job']==job,'Unreconciled release')
            C.require(C.read(folder/'ALLOCATION.json')['Account']=='mit_amf_advanced_cpu','Original allocation identity')
            jobs.add(str(job))
    C.require(str(row['original_job']) in jobs,'Original assignment absent')
    live=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],text=True)
    C.require(not any(j.split('_')[0] in jobs for j in live.splitlines()),'Original assignment live')
    for phase in ('round2_20261109_v1','round3_20261110_v1'):
        C.require(not (P/phase/'numerical_claims'/(row['key']+'.json')).exists(),'Unexpected numerical claim')
    return sorted(jobs)

def qualification(key):
    C.require(key in scope.TARGETS,'Unauthorized target')
    folder,expected_source=scope.qualification_location(key)
    qpath=folder/'QUALIFIED_PASS.json';q=C.read(qpath)
    complete=C.read(folder/'COMPLETE.json')
    C.require(complete['passed'] and complete['source']==expected_source and complete['qualified_pass_sha256']==C.sha(qpath),'Incomplete lifecycle')
    expected=folder/'verified_core.zip'
    C.require(q['source']==expected_source and Path(q['archive'])==expected,'Wrong qualified source/path')
    return qpath,q,expected

def run(key):
    C.require(key in scope.TARGETS and os.environ.get('SLURM_JOB_ID') and C.ROOT==C.CODE/'output','Publication scope/compute')
    C.verify_source()
    fields=dict(x.split('=',1) for x in subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True).split() if '=' in x)
    for k,v in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='1',Requeue='0').items():C.require(fields.get(k)==v,'Publication allocation '+k)
    C.require(fields['MinMemoryNode'] in ('2G','2048M'),'Publication memory')
    auth=C.read(C.CODE/'AUTHORIZATION.json');C.require(auth['approved_by_user'] and auth['keys']==scope.TARGETS and auth['publication'] and auth['allow_core229'] and auth['reference_attempts_per_core']==1,'User authorization differs')
    scope.source_proof()
    C.require(C.sha(R.AUDIT/'SCOPE.json')==R.SCOPEPIN,'Held classification changed')
    keys=[key]
    out=C.ROOT/'published'/key
    with contextlib.ExitStack() as locks:
        for f in [C.ROOT/'publication.lock',P/'control.lock',P/'round2_20261109_v1/solve/control.lock',P/'round3_20261110_v1/solve/control.lock']:locks.enter_context(C.locked(f))
        stopped()
        if (out/'PASS.json').exists():
            d=C.read(out/'PASS.json');C.require(d['passed'] and d['source']==C.sha(C.CODE/'LOCK.json'),'Prior publication conflicts');print('ALREADY_PUBLISHED');return
        targets=[];preserved={};protected_originals={};assignment_jobs={}
        for key in keys:
            row=next(r for r in C.read(R.AUDIT/'SCOPE.json')['cores'] if r['key']==key)
            record=C.read(Path(row['original_work'])/'input/record.json')
            C.require(C.key(record)==key and C.digest(record)==row['record_sha256'],'Original input identity')
            plan=C.read(P/'plans'/(record['case_id']+'.json'))
            C.require(next(r for r in plan['records'] if r['slot']==record['slot'])==record,'Production assignment differs')
            probe,_=scope.probe_proof(key)
            proofpath=probe/'PROTECTED.json'
            protected_originals.update(C.read(proofpath))
            R.verify_protection(protected_originals,C.read(R.AUDIT/'PROTECTED.json'))
            assignment_jobs[key]=reconcile(row)
            qpath,q,private=qualification(key);verify_archive(private,record,q)
            target=P/'results'/record['case_id']/f"core{record['slot']:03d}.zip"
            done=P/'runs'/record['case_id']/'done'/f"core{record['slot']:03d}.json"
            intent=dict(key=key,source=C.sha(C.CODE/'LOCK.json'),qualification_sha256=C.sha(qpath),archive_sha256=q['receipt']['archive_sha256'],target=str(target),record_sha256=C.digest(record))
            ip=out/'intents'/(key+'.json')
            if target.exists() or done.exists():
                C.require(ip.exists() and C.read(ip)==intent,'Existing result has no matching publication intent')
                C.require(target.exists() and C.sha(target)==intent['archive_sha256'],'Interrupted publication conflict')
                if not done.exists():C.write(done,q['receipt'])
            missing,fps=case_inventory(record['case_id'])
            before=out/'before'/(key+'.json')
            if before.exists():
                snapshot=C.read(before)
                C.require(set(missing) in (set(snapshot['missing']),set(snapshot['missing'])-{record['slot']}),'Case changed since publication intent')
                for p,fp in snapshot['preserved'].items():C.require(fingerprint(Path(p))==fp,'Sibling changed during interrupted publication')
                preserved.update(snapshot['preserved'])
            else:
                C.write(before,dict(missing=missing,preserved=fps));preserved.update(fps)
            C.write(ip,intent)
            targets.append((record,q,private,target,done))
        C.write(out/'RECONCILED.json',dict(assignment_jobs=assignment_jobs,production_states=[C.read(p) for p,_ in STATES]))
        completed=[];results=[]
        for record,q,private,target,done in targets:
            stopped();receipt=promote(private,target,record,q);C.write(done,receipt)
            missing,_=case_inventory(record['case_id'],full_hash=False)
            if not missing:
                C.write(P/'complete_cases'/(record['case_id']+'.json'),dict(case_id=record['case_id'],positions=522,plan_sha256=C.sha(P/'plans'/(record['case_id']+'.json')),method=METHOD))
                completed.append(record['case_id'])
            results.append(dict(key=C.key(record),archive=str(target),receipt=receipt,remaining_case_slots=missing))
        for p,fp in preserved.items():C.require(fingerprint(Path(p))==fp,'Existing sibling artifact changed')
        R.verify_protection(protected_originals,{})
        C.verify_source();stopped()
        C.write(out/'PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],keys=keys,
            new_solves=0,results=results,new_complete_cases=completed,total_full_cases=sum(1 for _ in (P/'complete_cases').glob('*.json')),
            original_files_preserved=len(protected_originals),existing_sibling_artifacts_preserved=len(preserved)))
        print('PUBLISHED',keys,'NEW_COMPLETE_CASES',completed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('key');run(p.parse_args().key)
