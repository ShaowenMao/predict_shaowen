"""Fixed precision-plateau scope; no scans, automatic retries or other holds."""
from pathlib import Path
import subprocess,tomllib
import common as C
from primary_proof import passing_primary
from accounting import terminal_jobs

BASE=Path('/orcd/data/juanes/001/shaowen')
OLD=BASE/'phase1_solver_targeted_20261110_v1'
DIAG=BASE/'phase1_direct_saved_20261111_v1'
PILOT=BASE/'phase1_precision_recovery_20261111_v1'
OLD_PIN='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502'
DIAG_PIN='9d7a36a4c92579697d06ac5962a2bd300ef696a719a53ecff7a49d92535663be'
PILOT_PIN='1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82'
PARENT=BASE/'phase1_direct_recovery_20261111_v2'
PARENT_PIN='2424a21d7aff1818b4341d2001be8cf4bfdc361be1c4cd3e97f3b92f1ddae5e5'
SAVED_SOURCES=[(BASE/'phase1_precision_reference5_20261111_v1','2e13b97f38cc46af649e3e73e63791cac4075e32216ee4e33b22da41acdd2727'),
    (BASE/'phase1_remaining56_tests_20261111_v1','aea89e68590c31f60d060a40c009b82ee76df0a2983d7fa97a9da0af4e44463a')]
PILOT_KEYS=['s04_c002_pc04_ca40_IND_001_core253','s01_c002_pc04_ca40_IND_008_core283','s04_c005_pc06_ca40_IND_001_core422']
SCOPE=C.read(C.CODE/'SCOPE.json')
TARGETS=SCOPE['keys']
NEW_KEYS=SCOPE['new_keys']

def batch_keys(batch):
    if batch=='pilot':return PILOT_KEYS
    C.require(batch in [str(i) for i in range(len(SCOPE['batches']))],'Invalid finite batch')
    return SCOPE['batches'][int(batch)]

def source_proof():
    C.verify_source()
    for root,pin in [(OLD,OLD_PIN),(DIAG,DIAG_PIN),(PILOT,PILOT_PIN),(PARENT,PARENT_PIN)]+SAVED_SOURCES:
        C.require(C.sha(root/'LOCK.json')==pin,'Qualified dependency pin changed')
        for n,h in C.read(root/'LOCK.json')['code'].items():C.require(C.sha(root/n)==h,'Qualified source changed')
    close=C.read(PILOT/'output/CLOSEOUT.json')
    C.require(close['passed'] and close['source']==PILOT_PIN and len(close['results'])==3,'Full-recovery pilot missing')
    C.require(C.sha(PILOT/'output/CLOSEOUT.json')==SCOPE['pilot_closeout_sha256'],'Pilot closeout changed')
    auth=C.read(C.CODE/'AUTHORIZATION.json')
    C.require(auth['approved_by_user'] and auth['keys']==TARGETS and auth['allow_core229'] and auth['reference_attempts_per_core']==1,'Fourth attempt not authorized')

def probe_proof(key):
    C.require(key in TARGETS,'Outside precision scope')
    source_proof()
    row=SCOPE['rows'][key]
    C.require(row['category']=='saved_primary_failed_reference' and row['prior_scientific_failures']==3,'Wrong failure class/count')
    for p,h in row['metadata'].items():C.require(C.sha(p)==h,'Saved failure evidence changed')
    probe=OLD/'output/probes'/key
    complete=C.read(probe/'DIAGNOSTIC_COMPLETE.json')
    for n,h in complete['files'].items():C.require(C.sha(probe/n)==h,'Saved probe artifact changed')
    failed=C.read(probe/'PROBE_RESULT.json')
    C.require(not failed['passed'] and failed['failed_stage'] in ('primary','reference'),'Original failing state missing')
    passing_primary(tomllib.loads(Path(row['saved_result']).read_text()),row)
    status=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(SCOPE['previous_jobs']),'--format=JobIDRaw,JobID%40,State'],text=True)
    terminal_jobs(status,SCOPE['previous_jobs'])
    previous=BASE/'phase1_final_reference4_20261111_v1'
    C.require(C.sha(previous/'LOCK.json')=='9e9c3a7ed5d1ebac3b94ce6af977baac55a0ee223e1d69135ddfd18eb1b66053','Preflight predecessor changed')
    error=C.read(previous/'output'/key/'ERROR.json')
    C.require('Previous diagnostic accounting missing' in error['error'],'Unclassified predecessor failure')
    C.require({p.name for p in (previous/'output'/key).iterdir()}=={'ERROR.json'},'Predecessor may have performed numerical work')
    status=subprocess.check_output(['sacct','-X','-n','-P','-j','28694786,28694787','--format=JobID%40,State'],text=True)
    rows=[x.split('|') for x in status.splitlines() if x.strip()]
    C.require({x[0] for x in rows}=={'28694786_'+str(i) for i in range(4)}|{'28694787'}
        and all(x[1] in ('FAILED','COMPLETED') for x in rows),'Preflight predecessor not fully terminal')
    return probe,complete

def qualification_location(key):
    C.require(key in TARGETS,'Outside publication scope')
    root,pin=(PILOT/'output',PILOT_PIN) if key in PILOT_KEYS else (C.ROOT,C.sha(C.CODE/'LOCK.json'))
    return root/key,pin
