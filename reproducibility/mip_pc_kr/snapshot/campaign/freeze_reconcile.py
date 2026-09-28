from common_launch import *

ROOT.mkdir(parents=True,exist_ok=True)
for n in ('logs','audit','submissions','batches','accepted','holds','prepared','claims','case_plans','resources'):
    (ROOT/n).mkdir(exist_ok=True)
files={n:sha(HERE/n) for n in ('common_launch.py','reconcile.py','reconcile.sh','freeze_reconcile.py')}
external={str(p):sha(p) for p in (REPLAY/'source_lock.json',REPLAY/'spec.json',REPLAY/'campaign.json',
          NUM/'source_lock.json',QUAL/'resource.py',QUAL/'COMPLETE_REVIEW.json',Q3/'Kernel.jl',Q3/'SelectStates.jl')}
lock=dict(files=files,external=external,authorized_full_campaign=True,reservoir_submission=False,
          authorization='User explicitly authorized reconcile, launch checks, progressive ramp and full WRR60 Pc100/Kr10 campaign in this turn.',
          old_campaign_read_only=True,production_read_only=True)
p=HERE/'LOCK.json';assert not p.exists();p.write_text(json.dumps(lock,indent=2,sort_keys=True)+'\n')
write(ROOT/'AUTHORIZATION.json',dict(**lock,release=str(HERE),lock_sha256=sha(p)))
print('RECONCILIATION_RELEASE_FROZEN',sha(p))
