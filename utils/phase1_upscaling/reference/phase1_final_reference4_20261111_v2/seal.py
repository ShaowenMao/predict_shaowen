import common as C
C.require(not (C.CODE/'LOCK.json').exists(),'Never reseal')
parent=C.CODE.parent/'phase1_direct_saved_20261111_v1'
C.require(C.sha(parent/'LOCK.json')=='9d7a36a4c92579697d06ac5962a2bd300ef696a719a53ecff7a49d92535663be','Direct source')
for n in ['DirectAudit.jl','AccurateAudit.jl']:
    C.require(C.sha(C.CODE/n)==C.sha(parent/n),'Direct arithmetic changed')
recovery=C.CODE.parent/'phase1_direct_recovery_20261111_v2'
for n in ('PrecisionRecovery.jl','StrictProbe.jl','ResumeContract.jl'):
    C.require(C.sha(C.CODE/n)==C.sha(recovery/n),'Qualified complete recovery arithmetic changed')
files=[p for p in C.CODE.rglob('*') if p.is_file() and p.name!='HANDOFF.md' and not ({'results','output','__pycache__'} & set(p.relative_to(C.CODE).parts))]
C.write(C.CODE/'LOCK.json',dict(code={p.relative_to(C.CODE).as_posix():C.sha(p) for p in files},
    method='ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832',saved_diagnostic_solves_repeated=0))
print(C.sha(C.CODE/'LOCK.json'))
