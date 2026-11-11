"""Freeze once, preserving the already qualified numerical implementations."""
import common as C
import scope

C.require(not (C.CODE/'LOCK.json').exists(),'Never reseal a frozen candidate')
pilot=C.CODE.parent/'phase1_precision_recovery_20261111_v1'
diag=C.CODE.parent/'phase1_remaining56_tests_20261111_v1'
C.require(C.sha(pilot/'LOCK.json')==scope.PILOT_PIN and C.sha(diag/'LOCK.json')==scope.DIAG_PIN,'Parent pins')
unchanged=['AccurateAudit.jl','PrecisionRecovery.jl','StrictProbe.jl','ResumeContract.jl','recover_flow.jl','test_integration.jl']
for n in unchanged:C.require(C.sha(C.CODE/n)==C.sha(pilot/n),'Qualified numerical source modified: '+n)
C.require(C.sha(C.CODE/'experiment.jl')==C.sha(diag/'experiment.jl'),'Qualified precision diagnostic changed')
names=sorted(p.name for p in C.CODE.iterdir() if p.suffix in ('.py','.jl','.sh'))+['SCOPE.json','AUTHORIZATION.json']
C.write(C.CODE/'LOCK.json',dict(code={n:C.sha(C.CODE/n) for n in names},prior_full_recovery_source=scope.PILOT_PIN,
    numerical_sources_unchanged=unchanged,method='ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'))
print(C.sha(C.CODE/'LOCK.json'))
