"""Preserve pre-switch assignments; route only new numerical work to this runner."""
import os
from pathlib import Path
import sys
import common as C

def record():
    path = C.ROOT / 'CACHE_ADOPTION.json'
    if not path.exists():
        return None
    a = C.read(path)
    lock = C.read(C.CODE / 'LOCK.json')
    C.require(a['new_run'] == dict(code=str(C.CODE), source=C.sha(C.CODE / 'LOCK.json')), 'Adoption runner identity')
    C.require(a['old_run'] == dict(code=lock['legacy_code'], source=lock['legacy_source']), 'Adoption predecessor')
    C.require(a['scope_sha256'] == C.sha(C.ROOT / 'SCOPE.json'), 'Adoption scope changed')
    C.require(a['configuration'] == lock['configuration_sha256'] and a['method'] == lock['method_sha256'], 'Adoption science identity')
    C.require(a['approved_by_user'] is True, 'Transition not authorized')
    return a

def authorized_source(auth, scope):
    if auth['source_sha256'] == C.sha(C.CODE / 'LOCK.json'):
        return True
    a = record()
    return bool(a and auth['source_sha256'] == a['old_run']['source']
                and C.sha(C.ROOT / 'LAUNCH_AUTHORIZATION.json') == a['authorization_sha256'])

def legacy_assignment(case):
    a = record()
    if not a or case not in a['legacy_cases']:
        return None
    row = a['legacy_cases'][case]
    for name, h in row['files'].items():
        C.require(C.sha(C.ROOT / name) == h, 'Legacy assignment changed: ' + name)
    return row

def preserve_legacy_assignment(case):
    return legacy_assignment(case) is not None

def legacy_target(mode, case, array_job, job):
    row = legacy_assignment(case)
    if row is None:
        return None
    C.require(mode in ('worker', 'fixture'), 'Only numerical/fixture jobs may use legacy routing')
    C.require(str(array_job or job) == row['launch']['job'], 'Legacy worker job identity')
    a = record()
    old = Path(a['old_run']['code'])
    C.require(C.sha(old / 'LOCK.json') == a['old_run']['source'], 'Legacy source pin changed')
    for name, h in C.read(old / 'LOCK.json')['code'].items():
        C.require(C.sha(old / name) == h, 'Legacy source changed: ' + name)
    return old

def route_legacy_worker(mode, case):
    old = legacy_target(mode, case, os.environ.get('SLURM_ARRAY_JOB_ID'), os.environ.get('SLURM_JOB_ID'))
    if old is None:
        return
    print('PRESERVED_LEGACY_RUNNER', str(old), case, flush=True)
    python = '/usr/bin/python3.12'
    if mode == 'worker':
        args = [python, '-B', str(old / 'worker.py')] + sys.argv[1:]
    else:
        args = [python, '-B', '-c', 'import sys;sys.path.insert(0,sys.argv[1]);import control;control.fixture(sys.argv[2])', str(old), case]
    os.execv(python, args)
