"""Fixed precision-plateau scope; no scans, automatic retries or other holds."""
from pathlib import Path
import common as C

BASE=Path('/orcd/data/juanes/001/shaowen')
OLD=BASE/'phase1_solver_targeted_20261110_v1'
DIAG=BASE/'phase1_remaining56_tests_20261111_v1'
PILOT=BASE/'phase1_precision_recovery_20261111_v1'
OLD_PIN='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502'
DIAG_PIN='aea89e68590c31f60d060a40c009b82ee76df0a2983d7fa97a9da0af4e44463a'
PILOT_PIN='1bf8764d77764c1813cc6bbcc59911deb52c1fa26162b8b0cabb268f8bcc4b82'
PILOT_KEYS=['s04_c002_pc04_ca40_IND_001_core253','s01_c002_pc04_ca40_IND_008_core283','s04_c005_pc06_ca40_IND_001_core422']
SCOPE=C.read(C.CODE/'SCOPE.json')
TARGETS=SCOPE['keys']
NEW_KEYS=SCOPE['new_keys']

def batch_keys(batch):
    if batch=='pilot':return PILOT_KEYS
    C.require(batch in [str(i) for i in range(5)],'Invalid finite batch')
    i=int(batch);return NEW_KEYS[8*i:8*(i+1)]

def source_proof():
    C.verify_source()
    for root,pin in ((OLD,OLD_PIN),(DIAG,DIAG_PIN),(PILOT,PILOT_PIN)):
        C.require(C.sha(root/'LOCK.json')==pin,'Qualified dependency pin changed')
        for n,h in C.read(root/'LOCK.json')['code'].items():C.require(C.sha(root/n)==h,'Qualified source changed')
    close=C.read(PILOT/'output/CLOSEOUT.json')
    C.require(close['passed'] and close['source']==PILOT_PIN and len(close['results'])==3,'Full-recovery pilot missing')
    C.require(C.sha(PILOT/'output/CLOSEOUT.json')==SCOPE['pilot_closeout_sha256'],'Pilot closeout changed')

def probe_proof(key):
    C.require(key in TARGETS,'Outside precision scope')
    source_proof()
    row=SCOPE['rows'][key]
    C.require(row['category']=='stable_reference_plateau' and row['prior_scientific_failures']==2,'Wrong failure class/count')
    for p,h in row['metadata'].items():C.require(C.sha(p)==h,'Saved failure evidence changed')
    probe=OLD/'output/probes'/key
    complete=C.read(probe/'DIAGNOSTIC_COMPLETE.json')
    for n,h in complete['files'].items():C.require(C.sha(probe/n)==h,'Saved probe artifact changed')
    failed=C.read(probe/'PROBE_RESULT.json')
    C.require(not failed['passed'] and failed['failed_stage']=='reference','Not the original failed reference state')
    return probe,complete

def qualification_location(key):
    C.require(key in TARGETS,'Outside publication scope')
    root,pin=(PILOT/'output',PILOT_PIN) if key in PILOT_KEYS else (C.ROOT,C.sha(C.CODE/'LOCK.json'))
    return root/key,pin
