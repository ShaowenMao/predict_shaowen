"""Fixed precision-plateau scope; no scans, automatic retries or other holds."""
from pathlib import Path
import common as C

BASE=Path('/orcd/data/juanes/001/shaowen')
OLD=BASE/'phase1_solver_targeted_20261110_v1'
DIAG=BASE/'phase1_direct_saved_20261111_v1'
PILOT=BASE/'phase1_precision_recovery_20261111_v1'
OLD_PIN='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502'
DIAG_PIN='9d7a36a4c92579697d06ac5962a2bd300ef696a719a53ecff7a49d92535663be'
PILOT_PIN='1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82'
PARENT=BASE/'phase1_direct_recovery_20261111_v2'
PARENT_PIN='2424a21d7aff1818b4341d2001be8cf4bfdc361be1c4cd3e97f3b92f1ddae5e5'
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
    for root,pin in ((OLD,OLD_PIN),(DIAG,DIAG_PIN),(PILOT,PILOT_PIN),(PARENT,PARENT_PIN)):
        C.require(C.sha(root/'LOCK.json')==pin,'Qualified dependency pin changed')
        for n,h in C.read(root/'LOCK.json')['code'].items():C.require(C.sha(root/n)==h,'Qualified source changed')
    close=C.read(PILOT/'output/CLOSEOUT.json')
    C.require(close['passed'] and close['source']==PILOT_PIN and len(close['results'])==3,'Full-recovery pilot missing')
    C.require(C.sha(PILOT/'output/CLOSEOUT.json')==SCOPE['pilot_closeout_sha256'],'Pilot closeout changed')

def probe_proof(key):
    C.require(key in TARGETS,'Outside precision scope')
    source_proof()
    row=SCOPE['rows'][key]
    C.require(row['category']=='remaining_primary_failure' and row['prior_scientific_failures']==2,'Wrong failure class/count')
    for p,h in row['metadata'].items():C.require(C.sha(p)==h,'Saved failure evidence changed')
    probe=OLD/'output/probes'/key
    complete=C.read(probe/'DIAGNOSTIC_COMPLETE.json')
    for n,h in complete['files'].items():C.require(C.sha(probe/n)==h,'Saved probe artifact changed')
    failed=C.read(probe/'PROBE_RESULT.json')
    C.require(not failed['passed'] and failed['failed_stage']=='primary','Not the original failed primary state')
    return probe,complete

def qualification_location(key):
    C.require(key in TARGETS,'Outside publication scope')
    root,pin=(PILOT/'output',PILOT_PIN) if key in PILOT_KEYS else (C.ROOT,C.sha(C.CODE/'LOCK.json'))
    return root/key,pin
