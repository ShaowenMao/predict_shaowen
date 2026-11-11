"""Fixed held-core scope, ordered for case completion; no scanning/redrawing."""
from collections import Counter
import common as C

FIRST=['s04_c005_pc04_ca40_IND_003_core345','s02_c024_pc06_ca40_IND_003_core380','s02_c024_pc06_ca40_IND_003_core402']
EXCLUDED={'s04_c005_pc04_ca40_IND_003_core182','s01_c005_pc04_ca40_IND_008_core378','s02_c024_pc06_ca40_IND_003_core371'}
ROWS=[r for r in C.read(C.CODE/'AUDIT_SCOPE.json')['cores'] if r['category'] in ('strict_solver_correction_failed','singular_factorization') and r['key'] not in EXCLUDED]
counts=Counter(r['case_id'] for r in ROWS)
TARGETS=FIRST+[r['key'] for r in sorted(ROWS,key=lambda r:(counts[r['case_id']],r['case_id'],r['slot'])) if r['key'] not in FIRST]
C.require(len(TARGETS)==len(set(TARGETS))==154,'Fixed remaining scope differs')

def batch_keys(number):
    C.require(type(number) is int and 0<=number<20,'Invalid finite batch')
    return FIRST if number==0 else TARGETS[3+8*(number-1):3+8*number]

def probe_proof(key):
    p=C.ROOT/'probes'/key;d=C.read(p/'DIAGNOSTIC_COMPLETE.json')
    C.require(d['passed'] and d['probe_result'] and d['source']==C.sha(C.CODE/'LOCK.json'),'Probe not passed')
    for n,h in d['files'].items():C.require(C.sha(p/n)==h,'Probe evidence changed: '+n)
    return p,d
