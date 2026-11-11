"""One attempt per fixed core; independent failures do not block other cores."""
import contextlib
import os
from pathlib import Path
import sys
import traceback
import common as C
import scope
import recovery as R
import probe
import publish as P

def preflight(key):
    C.verify_source();P.stopped()
    C.require(key in scope.TARGETS,'Outside authorization')
    row=next(r for r in scope.ROWS if r['key']==key)
    P.reconcile(row)
    C.require(row['unsuccessful_core_attempts']==1,'Prior attempts differ')
    C.require(not (R.P/'complete_cases'/(row['case_id']+'.json')).exists(),'Complete case excluded')
    for path in [R.P/'results'/row['case_id']/f"core{row['slot']:03d}.zip",R.P/'runs'/row['case_id']/'done'/f"core{row['slot']:03d}.json"]:
        C.require(not path.exists(),'Accepted or uncertain result excluded')
    for path in [C.ROOT/key,C.ROOT/'probes'/key,C.ROOT/'isolated'/(key+'.json')]:
        C.require(not path.exists(),'Previous attempt preserved; no automatic retry')
    return row

def worker(key):
    C.require(os.environ.get('SLURM_JOB_ID'),'Compute only')
    with C.locked(C.ROOT/'locks'/(key+'.lock')):
        with contextlib.ExitStack() as locks:
            for f in [R.P/'control.lock',R.P/'round2_20261109_v1/solve/control.lock',R.P/'round3_20261110_v1/solve/control.lock']:
                locks.enter_context(C.locked(f))
            row=preflight(key)
            claim=C.ROOT/'claims'/(key+'.json')
            C.require(not claim.exists(),'Existing reservation; reconcile without repeating')
            C.write(claim,dict(key=key,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],original_failed_attempts=1,new_recovery_attempt=1))
        try:
            probe.run(scope.TARGETS.index(key))
            scope.probe_proof(key)
            R.run(key)
        except Exception as e:
            C.write(C.ROOT/'isolated'/(key+'.json'),dict(key=key,source=C.sha(C.CODE/'LOCK.json'),job=os.environ['SLURM_JOB_ID'],error=str(e),traceback=traceback.format_exc(),automatic_retry=False,original_failed_attempts=1,new_recovery_attempt=1,saved_outputs_preserved=True))
            raise

def published_batch(number):
    C.verify_source();P.stopped()
    folder=C.ROOT/'batches'/f'{number:03d}'
    keys=scope.batch_keys(number)
    admitted=C.read(folder/'ADMITTED.json')
    C.require(admitted['keys']==keys and admitted['source']==C.sha(C.CODE/'LOCK.json'),'Batch assignment differs')
    completed=[];held=[]
    for key in keys:
        if (C.ROOT/key/'QUALIFIED_PASS.json').exists():
            P.run(key);completed.append(key)
        else:
            held.append(key)
    C.write(folder/'PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),keys=keys,published=completed,isolated=held,
        all_passed=not held,automatic_retry=False,full_cases=sum(1 for _ in (R.P/'complete_cases').glob('*.json'))))
    print('BATCH_RESULT',number,'PUBLISHED',len(completed),'ISOLATED',len(held),flush=True)

if __name__=='__main__':
    mode=sys.argv[1];number=int(sys.argv[2])
    if mode=='worker':worker(scope.batch_keys(number)[int(os.environ['SLURM_ARRAY_TASK_ID'])])
    elif mode=='publish':published_batch(number)
    else:raise RuntimeError('Unknown mode')
