"""Only three already-passing precision probes; no numerical probe retries."""
import tomllib
import common as C

BASE=C.Path('/orcd/data/juanes/001/shaowen')
OLD=BASE/'phase1_solver_targeted_20261110_v1'
DIAG=BASE/'phase1_remaining56_tests_20261111_v1'
OLD_PIN='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502'
DIAG_PIN='aea89e68590c31f60d060a40c009b82ee76df0a2983d7fa97a9da0af4e44463a'
TARGETS=['s04_c002_pc04_ca40_IND_001_core253',
         's01_c002_pc04_ca40_IND_008_core283',
         's04_c005_pc06_ca40_IND_001_core422']

def probe_proof(key):
    C.require(key in TARGETS,'Outside the three-core qualification')
    for root,pin in ((OLD,OLD_PIN),(DIAG,DIAG_PIN)):
        C.require(C.sha(root/'LOCK.json')==pin,'Predecessor source changed')
        for name,h in C.read(root/'LOCK.json')['code'].items():
            C.require(C.sha(root/name)==h,'Predecessor file changed: '+name)
    manifest=C.read(C.CODE/'SCOPE.json')['proofs'][key]
    for name,h in manifest.items():
        C.require(C.sha(DIAG/'output'/key/name)==h,'Saved precision proof changed: '+name)
    d=C.read(DIAG/'output'/key/'COMPLETE.json')
    C.require(d['passed'] and d['execution_complete'] and d['source']==DIAG_PIN,
              'Saved diagnostic not passed')
    C.require(d['production_writes']==d['completed_phase_calls_repeated']==0,'Diagnostic side effects')
    C.require(d['result']==result(key),'Diagnostic/result disagreement')
    probe=OLD/'output/probes'/key
    for name,h in C.read(probe/'DIAGNOSTIC_COMPLETE.json')['files'].items():
        C.require(C.sha(probe/name)==h,'Original probe evidence changed')
    return probe,d

def result(key):
    return tomllib.loads((DIAG/'output'/key/'RESULT.toml').read_text())
