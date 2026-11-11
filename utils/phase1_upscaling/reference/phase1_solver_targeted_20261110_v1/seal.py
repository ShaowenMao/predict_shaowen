import common as C
import scope
import recovery as R
C.require(C.sha(C.CODE/'AUDIT_SCOPE.json')==R.SCOPEPIN,'Original held scope changed')
names=sorted(p.name for p in C.CODE.iterdir() if p.suffix in ('.py','.jl','.sh'))+['AUDIT_SCOPE.json','QUALIFIED_378.json','QUALIFIED_371.json']
C.write(C.CODE/'LOCK.json',dict(code={n:C.sha(C.CODE/n) for n in names},method=R.METHOD,keys=scope.TARGETS,
    first=scope.FIRST,max_batch=8,completed_calls_must_not_repeat=True,automatic_retry=False,
    prior_recovery_preflight_failures=1,prior_audit_preflight_failures=1,
    parent='b81acc7d02ca3e2fceb4a010d7f7776fbf4c880fab1c1ba27a7a24d715b4e79b'))
C.write(C.CODE/'AUTHORIZATION.json',dict(approved_by_user=True,keys=scope.TARGETS,source=C.sha(C.CODE/'LOCK.json'),
    first_three_before_expansion=True,max_batch=8,unchanged_scientific_checks=True,no_completed_recalculation=True))
print(C.sha(C.CODE/'LOCK.json'))
