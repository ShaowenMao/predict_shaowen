"""Publish exactly sixteen qualified saved results. No exporter or solver calls."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import unittest
import zipfile
import common as C

BASE = Path('/orcd/data/juanes/001/shaowen')
P = BASE/'phase1_pc40_adaptive10_campaign_20261030_v1'
A = BASE/'phase1_held173_audit_20261110_v1/output'
QCODE = BASE/'phase1_exact_spacing_recovery_20261110_v1'
Q = QCODE/'output'
HOME = BASE/'phase1_saved16_closeout_20261110_v1/output'
PARENT_PIN = '5bcedae1b145ac77f0fb19975acc8ed7addcae4538c3d5ba546caea3d255bbd6'
METHOD = 'ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
CONFIG = '7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36'
AUDIT_PASS = 'a3916a3497504a0af2e2d7271de4ee9323f50447f0badce0a452cb2effe2cf7d'
QPIN = '67fda590f27d0ffac6ebaac334b6ed5e62746cda7f1b97ef1b48ac4d6b48dfe3'
QPASS = 'd782621beecd71c3ad8981391237704ac5391bcd64fc468e8b997565004ae3cb'
POLICY = 'exact_positive_saturation_nodes_native_validated_v1'
TERMINAL = {'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE','REVOKED'}


def authorize():
    C.verify_source()
    C.require(C.ROOT == HOME, 'Wrong output directory')
    C.require('SLURM_JOB_ID' in os.environ, 'Compute job required')
    C.require(C.sha(HOME.parent/'LOCK.json') == PARENT_PIN, 'Qualified parent operator changed')
    a = C.read(HOME.parent/'AUTHORIZATION.json')
    C.require(a['approved_by_user'] is True and a['source'] == PARENT_PIN, 'Not authorized')
    C.require(a['method'] == METHOD and a['policy'] == POLICY and a['new_solves'] == 0, 'Authorization scope')
    C.require(len(a['keys']) == len(set(a['keys'])) == 16, 'Authorization key scope')
    C.require(C.sha(A/'PASS.json') == AUDIT_PASS and C.sha(Q/'QUALIFIED_PASS.json') == QPASS, 'Prior qualification changed')
    C.require(C.sha(QCODE/'LOCK.json') == QPIN, 'Qualified export source changed')
    for name,h in C.read(QCODE/'LOCK.json')['code'].items():
        C.require(C.sha(QCODE/name) == h, 'Qualified export implementation changed')
    for path, expected in [(P/'STATE.json',dict(active=[],cursor=2592)),(P/'round2_20261109_v1/solve/STATE.json',dict(active=[],cursor=125)),(P/'round3_20261110_v1/solve/STATE.json',dict(active=[],cursor=55))]:
        C.require(C.read(path) == expected, 'Campaign resumed; reconcile before publication')
    # Check only this campaign's recorded stages; unrelated shared jobs are irrelevant.
    C.require((P/'FIRST_PASS_TERMINAL.json').is_file() and (P/'round2_20261109_v1/ROUND_TWO_TERMINAL.json').is_file() and (P/'round3_20261110_v1/ROUND_THREE_TERMINAL.json').is_file(), 'Missing campaign terminal evidence')
    scope = C.read(A/'SCOPE.json')
    C.require(C.sha(A/'SCOPE.json') == C.read(A/'PASS.json')['scope_sha256'], 'Classification scope changed')
    rows = [r for r in scope['cores'] if r['category'] == 'export_minimum_saturation_spacing']
    C.require(sorted(r['key'] for r in rows) == sorted(a['keys']), 'Targets differ from approved sixteen')
    q = C.read(Q/'QUALIFIED_PASS.json')
    C.require(q['passed'] and q['new_solves'] == 0 and q['cores'] == 16 and q['source'] == QPIN and q['policy'] == POLICY and q['raw_values_unchanged'], 'Export qualification incomplete')
    C.require(sorted(r['key'] for r in q['results']) == sorted(a['keys']), 'Qualification scope')
    return a,rows,{r['key']:r for r in q['results']}


def fingerprints(path):
    C.require(path.is_file() and not path.is_symlink(), 'Unsafe archive')
    s = path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def verify_case(root, case, missing=None, protected=None):
    plan_path=root/'plans'/(case+'.json');plan=C.read(plan_path)
    records=plan['records']
    C.require(len(records)==522 and {r['slot'] for r in records}==set(range(1,523)), 'Case coverage incomplete')
    for r in records:
        C.require(r['case_id']==case and r['configuration_sha256']==CONFIG, 'Wrong case configuration')
        if r['slot']==missing:
            continue
        receipt_path=root/'runs'/case/'done'/f"core{r['slot']:03d}.json"
        receipt=C.read(receipt_path)
        C.require(receipt['key']==C.key(r) and receipt['record_sha256']==C.digest(r), 'Receipt identity')
        C.require(receipt['scientific_pass'] and receipt['method']==METHOD, 'Unaccepted case result')
        archive=root/'results'/case/f"core{r['slot']:03d}.zip"
        fp=fingerprints(archive)
        C.require(fp[2]==receipt['bytes'], 'Missing or truncated archive')
        if protected is not None:
            protected['files'][str(receipt_path)]=C.sha(receipt_path)
            protected['archives'][str(archive)]=fp
    return plan


def check_preserved(protected):
    for f,h in protected['files'].items():C.require(C.sha(f)==h,'Preserved evidence changed: '+f)
    # st_dev identifies a node's mount, not the shared file. Retain it as provenance
    # in the original snapshot, but compare inode, bytes and mtime across nodes.
    for f,fp in protected['archives'].items():C.require(fingerprints(Path(f))[1:]==fp[1:],'Prior accepted archive changed: '+f)


def reconcile_assignments(rows, protected):
    assigned=set()
    for row in rows:
        case=row['case_id']
        for root,label in [(P,'num_'+case),(P/'round2_20261109_v1/finish','r2f_'+case)]:
            launch=root/'cases'/case/'LAUNCHED.json'
            terminal=root/'cases'/case/'TERMINAL.json'
            d=C.read(launch);t=C.read(terminal)
            C.require(d['job']==t['job'] and all(v[0] in TERMINAL for v in t['tasks'].values()),'Assignment not terminal')
            for suffix,job in [('',d['job']),('_gate',d['gate'])]:
                folder=root/'submissions'/(label+suffix)
                ack=C.read(folder/'ACK.json');intent=C.read(folder/'INTENT.json')
                C.require(ack['job']==job and ack['spec']==intent['spec'],'Unreconciled submission identity')
                for name in ('ACK.json','INTENT.json','ALLOCATION.json','RELEASED.json'):
                    protected['files'][str(folder/name)]=C.sha(folder/name)
                assigned.add(str(job))
            protected['files'][str(launch)]=C.sha(launch)
            protected['files'][str(terminal)]=C.sha(terminal)
    queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i'],text=True)
    C.require(not any(x.strip().split('_')[0] in assigned for x in queue.splitlines()),'Original assignment still active')
    return sorted(assigned)


def validate_saved(row, q, protected):
    key=row['key'];original=Path(row['original_work']);private=Path(q['work'])
    C.require(private == Q/'private'/key, 'Unexpected qualified payload path')
    record=C.read(original/'input/record.json')
    C.require(C.key(record)==key and C.digest(record)==row['record_sha256'], 'Original saved record identity')
    C.require(C.read(private/'input/record.json')==record, 'Qualified input mismatch')
    proof=tomllib.loads((original/'flow/CORE_PASS.toml').read_text())
    C.require(proof['passed'] and proof['method']==METHOD and proof['record_sha256']==C.digest(record), 'Saved flow not verified')
    C.require(proof==tomllib.loads((private/'flow/CORE_PASS.toml').read_text()), 'Qualified flow proof differs')
    C.require(proof['input']==tomllib.loads((original/'input/INPUT_PASS.toml').read_text()), 'Original input proof differs')
    for name,h in proof['artifacts'].items():
        C.require(C.sha(original/'flow'/name)==h==C.sha(private/'flow'/name), 'Saved scientific artifact changed')
    for folder in ('input','flow'):
        original_members={f.relative_to(original).as_posix() for f in (original/folder).rglob('*') if f.is_file()}
        private_members={f.relative_to(private).as_posix() for f in (private/folder).rglob('*') if f.is_file()}
        C.require(original_members==private_members, 'Saved member inventory differs')
        for name in original_members:C.require(C.sha(original/name)==C.sha(private/name), 'Saved bytes changed')
    native=tomllib.loads((private/'NATIVE_PASS.toml').read_text())
    ex=C.read(private/'export/EXPORT_PASS.json')
    C.require(C.sha(private/'NATIVE_PASS.toml')==q['native_sha256'], 'Qualified native receipt changed')
    C.require(native['passed'] and native['method']==METHOD and native['record_sha256']==row['record_sha256'], 'Native acceptance missing')
    C.require(ex['passed'] and ex['export_policy']==POLICY and ex['export_source']==QPIN and ex['raw_curve_values_unchanged'] and ex['all_distinct_positive_saturation_nodes_preserved'], 'Unqualified export')
    C.require(C.sha(private/'export/sgof.csv')==native['sgof_sha256']==ex['sha256']==q['sgof_sha256'], 'Native/export table mismatch')
    for f in private.rglob('*'):
        if f.is_file():protected['files'][str(f)]=C.sha(f)
    for phase in ('round2_20261109_v1','round3_20261110_v1'):
        C.require(not (P/phase/'numerical_claims'/(key+'.json')).exists(), 'Unexpected numerical reservation; reconcile')
    case=row['case_id'];term=P/'cases'/case/'TERMINAL.json';d=C.read(term)
    C.require(d['job']==row['original_job'] and all(v[0] in TERMINAL for v in d['tasks'].values()), 'Original task not terminal')
    protected['files'][str(term)]=C.sha(term)
    term=P/'round2_20261109_v1/finish/cases'/case/'TERMINAL.json';d=C.read(term)
    C.require(all(v[0] in TERMINAL for v in d['tasks'].values()), 'Finish task not terminal')
    protected['files'][str(term)]=C.sha(term)
    for path in [P/'round2_20261109_v1/finish/plans'/(case+'.json'), P/'round2_20261109_v1/finish/plans'/(case+'.seal.json'),P/'round2_20261109_v1/finish/cases'/case/'LAUNCHED.json']:
        protected['files'][str(path)]=C.sha(path)
    return record


def promote_archive(private, target, record, adoption):
    C.verify_archive(private, record, METHOD)
    with zipfile.ZipFile(private) as z:
        C.require(json.loads(z.read('EXPORT_ADOPTION.json'))==adoption, 'Private archive adoption mismatch')
    expected=C.sha(private)
    if target.exists():
        C.require(C.sha(target)==expected, 'Existing canonical archive differs; never overwrite')
    else:
        # Atomic publication of already verified immutable bytes, not a new solve.
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(private,target)
    C.require(C.sha(target)==expected, 'Published archive checksum mismatch')
    return C.archive_receipt(target,record,METHOD)


def qualify():
    a,rows,q=authorize()
    C.require(not (C.ROOT/'QUALIFIED_PASS.json').exists(), 'Completed qualification must not repeat')
    suite=unittest.defaultTestLoader.discover(str(C.CODE),pattern='test_closeout.py')
    test=unittest.TextTestRunner(verbosity=2).run(suite)
    C.require(test.wasSuccessful(), 'Closeout regression failed')
    protected=dict(files={},archives={})
    # Preserve the 157 excluded solver holds and all original target evidence.
    evidence=C.read(A/'PROTECTED.json')
    originals=[r['original_work']+'/' for r in C.read(A/'SCOPE.json')['cores']]
    for f,h in evidence.items():
        if any(f.startswith(prefix) for prefix in originals):protected['files'][f]=h
    protected['files'].update(C.read(Q/'PROTECTED.json'))
    check_preserved(protected)
    records={};plans={}
    for row in rows:
        r=validate_saved(row,q[row['key']],protected)
        plans[row['case_id']]=verify_case(P,row['case_id'],missing=row['slot'],protected=protected)
        C.require(next(x for x in plans[row['case_id']]['records'] if x['slot']==row['slot'])==r, 'Fixed production assignment differs')
        target=P/'results'/row['case_id']/f"core{row['slot']:03d}.zip"
        receipt=P/'runs'/row['case_id']/'done'/f"core{row['slot']:03d}.json"
        C.require(not target.exists() and not receipt.exists() and not (P/'complete_cases'/(row['case_id']+'.json')).exists(), 'Target already completed; stop')
        records[row['key']]=r
        protected['files'][str(P/'plans'/(row['case_id']+'.json'))]=C.sha(P/'plans'/(row['case_id']+'.json'))
    C.require(len(protected['archives'])==16*521, 'Expected 521 accepted siblings per target')
    jobs=reconcile_assignments(rows,protected)
    C.write(C.ROOT/'RECONCILED_JOBS.json',dict(job_ids=jobs,all_recorded_tasks_terminal=True,live_overlap=False))
    C.write(C.ROOT/'PROTECTED.json',protected)
    C.write(C.ROOT/'RECORDS.json',records)
    adoption=dict(policy=POLICY,method=METHOD,source=C.sha(C.CODE/'LOCK.json'),authorization_sha256=C.sha(C.CODE/'AUTHORIZATION.json'),export_source=QPIN,export_qualification_sha256=QPASS,classification_pass_sha256=AUDIT_PASS,scope=a['keys'],raw_curve_values_preserved=True,new_solves=0)
    C.write(C.ROOT/'ADOPTION.json',adoption)
    archives={}
    for row in rows:
        key=row['key'];work=C.ROOT/'payloads'/key
        C.require(not work.exists(), 'Uncertain prior payload; reconcile rather than retry')
        shutil.copytree(Path(q[key]['work']),work)
        C.write(work/'EXPORT_ADOPTION.json',adoption)
        target=C.ROOT/'archives'/(key+'.zip')
        receipt=C.publish(work,records[key],METHOD,target)
        C.require(C.verify_archive(target,records[key],METHOD)['scientific_pass'],'Archive lifecycle failure')
        archives[key]=dict(path=str(target),receipt=receipt)
    check_preserved(protected)
    C.verify_source()
    C.write(C.ROOT/'QUALIFIED_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),tests=test.testsRun,archives=archives,cores=16,new_solves=0,production_writes=0,existing_archives_preserved=len(protected['archives']),evidence_files_preserved=len(protected['files']),job=os.environ['SLURM_JOB_ID'],adoption_sha256=C.sha(C.ROOT/'ADOPTION.json')))
    print('ARCHIVE_LIFECYCLE_QUALIFIED 16',flush=True)


def close():
    a,rows,q=authorize()
    qual=C.read(C.ROOT/'QUALIFIED_PASS.json')
    C.require(qual['passed'] and qual['source']==PARENT_PIN and qual['cores']==16 and qual['new_solves']==0,'Archive lifecycle not qualified')
    guard=C.read(C.CODE/'qualification/GUARD_PASS.json')
    C.require(guard['passed'] and guard['source']==C.sha(C.CODE/'LOCK.json') and guard['parent_qualification_sha256']==C.sha(C.ROOT/'QUALIFIED_PASS.json'),'Portable guard not qualified')
    C.require(qual['adoption_sha256']==C.sha(C.ROOT/'ADOPTION.json'),'Adoption changed')
    with contextlib.ExitStack() as locks:
        for path in [C.ROOT/'closeout.lock',P/'control.lock',P/'round2_20261109_v1/solve/control.lock',P/'round3_20261110_v1/solve/control.lock']:
            locks.enter_context(C.locked(path))
        authorize()
        C.require(not (C.ROOT/'CLOSEOUT_PASS.json').exists(),'Closeout complete; do not repeat')
        protected=C.read(C.ROOT/'PROTECTED.json');check_preserved(protected)
        reconcile_assignments(rows,protected)
        records=C.read(C.ROOT/'RECORDS.json');adoption=C.read(C.ROOT/'ADOPTION.json')
        C.require(adoption['authorization_sha256']==C.sha(HOME.parent/'AUTHORIZATION.json'),'Authorization changed after qualification')
        done=[]
        for row in rows:
            key=row['key'];r=records[key]
            C.require(C.digest(r)==row['record_sha256'] and C.key(r)==key,'Target changed')
            target=P/'results'/r['case_id']/f"core{r['slot']:03d}.zip"
            receipt_path=P/'runs'/r['case_id']/'done'/f"core{r['slot']:03d}.json"
            item=qual['archives'][key];private=Path(item['path'])
            C.require(private==C.ROOT/'archives'/(key+'.zip') and C.sha(private)==item['receipt']['archive_sha256'],'Qualified archive changed')
            C.write(C.ROOT/'intents'/(key+'.json'),dict(key=key,record_sha256=C.digest(r),archive_sha256=C.sha(private),target=str(target),source=C.sha(C.CODE/'LOCK.json'),qualification_sha256=C.sha(C.ROOT/'QUALIFIED_PASS.json')))
            receipt=promote_archive(private,target,r,adoption)
            C.require(receipt==item['receipt'],'Published receipt differs from qualification')
            C.write(receipt_path,receipt)
            plan=verify_case(P,r['case_id'])
            C.write(P/'complete_cases'/(r['case_id']+'.json'),dict(case_id=r['case_id'],positions=522,plan_sha256=C.sha(P/'plans'/(r['case_id']+'.json')),method=METHOD))
            done.append(dict(key=key,case_id=r['case_id'],archive_sha256=receipt['archive_sha256']))
            print('CLOSED_CASE',r['case_id'],flush=True)
        check_preserved(protected)
        C.verify_source()
        count=sum(1 for _ in (P/'complete_cases').glob('*.json'))
        C.require(count==2505,'Unexpected completed-case count; reconcile')
        remaining=[r for r in C.read(A/'SCOPE.json')['cores'] if r['category']!='export_minimum_saturation_spacing']
        C.require(len(remaining)==157 and len({r['case_id'] for r in remaining})==87,'Remaining hold scope')
        for r in remaining:
            C.require(not (P/'results'/r['case_id']/f"core{r['slot']:03d}.zip").exists(),'Excluded held core changed')
            C.require(not (P/'runs'/r['case_id']/'done'/f"core{r['slot']:03d}.json").exists(),'Excluded held receipt changed')
        C.write(C.ROOT/'CLOSEOUT_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],new_solves=0,recovered_cores=16,new_full_cases=16,total_full_cases=count,total_cases=2592,remaining_held_cores=157,remaining_partial_cases=87,results=done,existing_archives_preserved=len(protected['archives']),protected_evidence_files=len(protected['files']),qualification_sha256=C.sha(C.ROOT/'QUALIFIED_PASS.json'),guard_qualification_sha256=C.sha(C.CODE/'qualification/GUARD_PASS.json'),adoption_sha256=C.sha(C.ROOT/'ADOPTION.json')))
        print('CLOSEOUT_PASS 16 CORES 2505 FULL_CASES',flush=True)


def qualify_guard():
    authorize()
    output=C.CODE/'qualification'
    C.require(not (output/'GUARD_PASS.json').exists(),'Guard already qualified; do not repeat')
    previous=C.read(C.ROOT/'QUALIFIED_PASS.json')
    C.require(previous['source']==PARENT_PIN and previous['passed'],'Parent archive lifecycle missing')
    old_log=C.ROOT/'logs/p1q_saved16_archive_closeout_28616793_4294967294.log'
    C.require('Prior accepted archive changed:' in old_log.read_text(),'Different prior failure')
    C.require(not (C.ROOT/'intents').exists(),'Prior publication began; reconcile before repair')
    suite=unittest.defaultTestLoader.discover(str(C.CODE),pattern='test_closeout.py')
    test=unittest.TextTestRunner(verbosity=2).run(suite)
    C.require(test.wasSuccessful(),'Portable guard regression')
    protected=C.read(C.ROOT/'PROTECTED.json')
    check_preserved(protected)
    paths=sorted(protected['archives'])
    first=Path(paths[0]);old=protected['archives'][paths[0]];new=fingerprints(first)
    case=first.parent.name;slot=int(first.stem[4:])
    record=next(r for r in C.read(P/'plans'/(case+'.json'))['records'] if r['slot']==slot)
    receipt=C.read(P/'runs'/case/'done'/(first.stem+'.json'))
    C.require(C.sha(first)==receipt['archive_sha256'],'First reported archive content actually changed')
    C.verify_archive(first,record,METHOD)
    for key,item in previous['archives'].items():
        r=C.read(C.ROOT/'RECORDS.json')[key]
        C.require(C.sha(Path(item['path']))==item['receipt']['archive_sha256'],'Qualified private archive changed')
        C.verify_archive(Path(item['path']),r,METHOD)
        C.require(not (P/'results'/r['case_id']/f"core{r['slot']:03d}.zip").exists(),'Unexpected publication')
    C.write(C.ROOT/'RECOVERY_ATTEMPTS.json',dict(condition='node_local_st_dev_compared_across_nodes',unsuccessful_publications=1,failed_job='28616793',failed_source=PARENT_PIN,failure_log_sha256=C.sha(old_log),previous_publication_started=False,original_export_failures_per_core=2,successful_export_qualification=True,numerical_solves=0))
    C.write(output/'GUARD_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),parent_qualification_sha256=C.sha(C.ROOT/'QUALIFIED_PASS.json'),tests=test.testsRun,job=os.environ['SLURM_JOB_ID'],existing_archives_preserved=len(protected['archives']),evidence_files_preserved=len(protected['files']),previous_fingerprint=old,current_fingerprint=new,reported_archive_sha256=C.sha(first),reported_archive_receipt_sha256=receipt['archive_sha256'],qualified_candidate_archives_verified=16,production_writes=0,new_solves=0))
    print('PORTABLE_GUARD_PASS',len(protected['archives']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['guard-qualify','closeout','recover']);args=parser.parse_args()
    if args.mode in ('guard-qualify','recover'):
        qualify_guard()
    if args.mode in ('closeout','recover'):
        close()
