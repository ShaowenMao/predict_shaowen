"""Finite compute-only tests and copied-checkpoint lifecycle; no numerical solves."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest
from unittest.mock import patch
import zipfile

import common as C
import worker

BASE=Path('/orcd/data/juanes/001/shaowen')
SEED=BASE/'phase1_pc40_adaptive10_runner_20261030_v4'
PRODUCTION=BASE/'phase1_pc40_adaptive10_campaign_20261030_v1'
SEED_PIN='709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8'
PARENT_PIN='c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf'


def main():
    C.require('SLURM_JOB_ID' in os.environ,'Compute allocation required')
    lock=C.verify_source();out=C.CODE/'qualification'
    C.require(C.ROOT==out and lock['round_two_only'],'Isolated test root required')
    C.require(not (out/'QUALIFIED_PASS.json').exists() and not (out/'FAILED.json').exists(),'Do not repeat a previous qualification body')
    C.require(C.sha(SEED/'LOCK.json')==SEED_PIN,'Qualification seed changed')
    C.require(C.read(SEED/'LOCK.json')['science']==lock['science'],'Saved scientific implementation differs')
    proof=C.read(SEED/'numerical/PACKET_QUALIFIED_PASS.json')
    C.require(proof['passed'] and proof['source']==SEED_PIN and proof['verified_cores']==32,'Seed packet not qualified')
    before_route=C.read(PRODUCTION/'RUN.json')
    C.require(before_route['source']==PARENT_PIN,'Production changed; re-assess test context')
    before_state=C.read(PRODUCTION/'STATE.json')
    suite=unittest.defaultTestLoader.loadTestsFromNames(['test_runner','test_cache_integration','test_timeout','test_recovery','test_allocation_guard','test_round3'])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    C.require(result.wasSuccessful() and not result.skipped,'Compute regression failure or missing Linux tests')
    C.write(out/'COMPUTE_UNIT_PASS.json',dict(passed=True,tests=result.testsRun,source=C.sha(C.CODE/'LOCK.json')))

    case=C.read(C.CODE/'STAGED.json')['pilot_cases'][0]
    plan=C.read(C.CODE/'plans'/f'{case}.json');records=plan['packets'][0]
    C.require(len(records)==32,'Fixture must cover a whole 32-core packet')
    root=out/'copied_packet';root.mkdir()
    C.write(root/'SCOPE.json',dict(purpose='qualification',cases=[case],packet_count=1))
    (root/'plans').mkdir();shutil.copyfile(C.CODE/'plans'/f'{case}.json',root/'plans'/f'{case}.json')
    hashes={};members={}
    for i,r in enumerate(records):
        source=SEED/'numerical/results'/case/f"core{r['slot']:03d}.zip"
        C.verify_archive(source,r,lock['method_sha256']);hashes[str(source)]=C.sha(source)
        with zipfile.ZipFile(source) as z:
            names=[n for n in z.namelist() if n.startswith(('input/','flow/')) or n=='SOLVE_STARTED.toml']
            members[r['slot']]={n:hashlib.sha256(z.read(n)).hexdigest() for n in names}
            if i==0:
                target=root/'results'/case/source.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
            else:
                work=root/'runs'/case/'cores'/C.key(r)
                for name in names:
                    p=work/name;p.parent.mkdir(parents=True,exist_ok=True)
                    with p.open('xb') as f:f.write(z.read(name))

    calls=[];original_process=worker.process
    def no_solves(args,log,fd,env=None,cwd=None,timeout=6900):
        C.require(log.name in ('runtime.log','export.log','native.log'),'Numerical/preparation launch forbidden in saved-flow qualification')
        calls.append(log.name)
        return original_process(args,log,fd,env,cwd,timeout)
    start=time.monotonic()
    with patch.object(C,'ROOT',root),patch.object(worker,'process',no_solves):
        worker.run(case,0,finish_only=True)
        first_seconds=time.monotonic()-start
        attempts=list((root/'runs'/case/'attempts').iterdir())
        C.require(len(attempts)==1,'Unexpected attempts')
        summary=C.read(attempts[0]/'SUMMARY.json')
        C.require(summary['verified']==32 and summary['checkpoint_only'] and not summary['flow_timed_out'],'Finish-only failed')
        before={r['slot']:C.sha(root/'results'/case/f"core{r['slot']:03d}.zip") for r in records}
        n_calls=len(calls)
        worker.run(case,0,finish_only=True)
        C.require(len(calls)==n_calls,'Accepted no-op launched a stage')
        for r in records:
            target=root/'results'/case/f"core{r['slot']:03d}.zip"
            C.verify_archive(target,r,lock['method_sha256'])
            C.require(C.sha(target)==before[r['slot']],'Accepted result changed')
            with zipfile.ZipFile(target) as z:
                for n,h in members[r['slot']].items():
                    C.require(hashlib.sha256(z.read(n)).hexdigest()==h,'Saved input/flow was altered')
        # Real checksum rejection, using a private copied checkpoint only.
        r=records[1];bad=root/'corrupt_copy'
        source=Path(next(p for p in hashes if p.endswith(f"core{r['slot']:03d}.zip")))
        with zipfile.ZipFile(source) as z:
            for name in members[r['slot']]:
                p=bad/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
        with (bad/'flow/kr.csv').open('ab') as f:f.write(b'corrupt fixture\n')
        try:worker.flow_proof(bad,r,lock['method_sha256'])
        except RuntimeError as e:C.require('checksum' in str(e),'Unexpected rejection')
        else:raise RuntimeError('Corrupted checkpoint accepted')

    C.require(first_seconds<600,'Measured finalization exceeds proposed ten-minute reserve')
    for p,h in hashes.items():C.require(C.sha(p)==h,'Original qualification archive changed')
    C.require(C.read(PRODUCTION/'RUN.json')==before_route,'Production route changed during qualification')
    from test_recovery import qualify_round2_finish
    qualify_round2_finish()
    C.verify_source()
    C.write(out/'QUALIFIED_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),
        method=lock['method_sha256'],configuration=lock['configuration_sha256'],compute_tests=result.testsRun,
        saved_flow_copies_finalized=31,accepted_copy_preserved=1,accepted_packet_noop=True,
        original_archives_unchanged=True,saved_input_and_flow_bytes_unchanged=True,
        corrupted_copy_rejected=True,finalization_seconds=first_seconds,reserved_finalization_seconds=600,
        launched_stages=calls,new_numerical_solves=0,production_credit=0,production_adopted=False,
        production_route=before_route,cursor_before=before_state['cursor'],
        cursor_after=C.read(PRODUCTION/'STATE.json')['cursor'],
        job=os.environ['SLURM_JOB_ID'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        coverage='Forced-timeout worker regressions, real Linux child termination, and real saved-flow export/native/archive lifecycle; not a new numerical run',
        original_archives=hashes))
    print('TIMEOUT_CANDIDATE_QUALIFIED_PASS',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        C.write(C.CODE/'qualification/FAILED.json',dict(error=traceback.format_exc(),job=os.environ.get('SLURM_JOB_ID'),production_adopted=False))
        raise
