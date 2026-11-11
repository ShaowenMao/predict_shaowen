"""Pin saved precision evidence and this isolated complete-recovery candidate."""
from pathlib import Path
import common as C
import scope
import recovery as R

C.require(not (C.CODE/'LOCK.json').exists(),'Never reseal')
diag=Path('D:/codex_gom/studies/phase1_remaining56_tests_20261111_v1')
C.require(C.sha(diag/'LOCK.json')==scope.DIAG_PIN,'Diagnostic source pin')
close=C.read(diag/'results/output/CLOSEOUT.json')
names=['RESULT.toml','COMPLETE.json','PROTECTED.json','primary_1.0e-14_pressure.jls','reference_1.0e-13_pressure.jls']
proofs={}
for key in scope.TARGETS:
    p=diag/'results/output'/key
    r=next(x for x in close['tests'] if x['key']==key)
    C.require(r['passed'] and C.sha(p/'RESULT.toml')==r['result_sha256'] and
              C.sha(p/'COMPLETE.json')==r['receipt_sha256'],'Closeout/result mismatch')
    R.check_probe(C.read(p/'COMPLETE.json')['result'])
    proofs[key]={n:C.sha(p/n) for n in names}
C.write(C.CODE/'SCOPE.json',dict(keys=scope.TARGETS,proofs=proofs,diagnostic_source=scope.DIAG_PIN,
    diagnostic_closeout_sha256=C.sha(diag/'results/output/CLOSEOUT.json'),production_publication=False,
    prior_scientific_failures_per_core=2,precision_diagnostic_passed=True,full_recovery_attempt=1))
names=sorted(p.name for p in C.CODE.iterdir() if p.suffix in ('.py','.jl','.sh'))+['SCOPE.json']
C.write(C.CODE/'LOCK.json',dict(code={n:C.sha(C.CODE/n) for n in names},qualification_only=True,
    method=R.METHOD,science_source=R.R3PIN,precision_diagnostic_source=scope.DIAG_PIN))
print(C.sha(C.CODE/'LOCK.json'))
