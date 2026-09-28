"""Finite DAG dispatcher, deliberately restricted to one qualification fault."""
from contextlib import contextmanager
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from common import HERE, read, write, sha, verify, receipt
from resource import parse_allocation

ACCOUNT = 'mit_amf_advanced_cpu'


def memory_mib(value):
    m = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', value)
    if not m:
        raise ValueError('Unrecognized scheduler memory: ' + value)
    factors = {'': 1, 'K': Decimal(1)/1024, 'M': 1, 'G': 1024, 'T': 1024**2}
    return int((Decimal(m[1]) * factors[m[2]]).to_integral_value(rounding='ROUND_CEILING'))


def admission(rows, own_ids, add_cpus, add_mib, cap_cpus=500, cap_mib=1984*1024):
    cpus = mib = 0
    deferred = []
    seen = set()
    for job, state, cpu, mem, reason, account, qos in rows:
        assert job not in seen, 'Duplicate scheduler record'
        seen.add(job)
        if account != ACCOUNT:
            continue
        assert qos == ACCOUNT, 'Unexpected account/QoS pairing'
        assert state in ('RUNNING', 'PENDING', 'COMPLETING', 'CONFIGURING', 'SUSPENDED'), state
        if state == 'PENDING' and job not in own_ids and reason in ('Dependency', 'JobHeldUser', 'DependencyNeverSatisfied'):
            deferred.append(dict(job=job, reason=reason))
            continue
        cpus += int(cpu)
        mib += memory_mib(mem)
    assert min(cpus, mib, add_cpus, add_mib) >= 0
    return dict(passed=cpus+add_cpus <= cap_cpus and mib+add_mib <= cap_mib,
                used_cpus=cpus, used_mib=mib, add_cpus=add_cpus, add_mib=add_mib,
                external_dependency_or_user_held=deferred,
                policy='all_live_and_eligible_external_pending_plus_all_own_pending')


@contextmanager
def single_writer(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def dag():
    d = dict(audit=dict(cpus=1, memory_mib=1024, minutes=60, deps=[]),
             warm=dict(cpus=2, memory_mib=2816, minutes=30, deps=['audit']))
    if not (HERE/'receipts/audit.json').exists():
        return d
    receipt('audit')
    p = read(HERE/'audit/PLAN.json')
    for i, batch in enumerate(p['batches']):
        name = f'derive{i:03d}'
        if i == 0:
            d['prefix'] = dict(cpus=1, memory_mib=512, minutes=20, deps=['audit'])
            deps = ['prefix']
        else:
            deps = ['audit']
        d[name] = dict(cpus=1, memory_mib=512, minutes=20, deps=deps)
    derivatives = [k for k in d if k.startswith('derive')]
    d['assemble'] = dict(cpus=2, memory_mib=4096, minutes=60, deps=derivatives)
    d['native'] = dict(cpus=1, memory_mib=3072, minutes=45, deps=['assemble'])
    d['fullgrid'] = dict(cpus=2, memory_mib=12416, minutes=90, deps=['assemble'])
    d['final'] = dict(cpus=1, memory_mib=512, minutes=20, deps=['warm', 'native', 'fullgrid'])
    return d


def token(stage):
    return 'kr10q:' + sha(HERE/'LOCK.json')[:16] + ':' + stage


def returned_job(stage, scheduler=None):
    """An unresolved submission intent always holds; it never causes a duplicate."""
    folder = HERE/'intents'
    returned = folder/(stage+'.returned.json')
    if returned.exists():
        r = read(returned)
        assert r['token'] == token(stage) and str(r['job']).isdigit()
        return r['job']
    if not (folder/(stage+'.intent.json')).exists():
        return None
    # A lost sbatch response can be recovered only by a unique scheduler token.
    if scheduler is None:
        s = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%i|%k'], text=True, timeout=30)
        scheduler = [line.split('|', 1) for line in s.splitlines()]
    matches = {job for job, comment in scheduler if comment == token(stage)}
    assert len(matches) == 1, 'Uncertain submission: reconcile scheduler/accounting for ' + stage
    job = matches.pop()
    assert job.isdigit()
    write(returned, dict(job=job, token=token(stage), recovered_from_scheduler=True))
    return job


def queue():
    s = subprocess.check_output(['squeue', '-h', '-r', '-u', os.environ['USER'],
                                 '-o', '%i|%T|%C|%m|%r|%a|%q'], text=True, timeout=30)
    rows = [line.strip().split('|') for line in s.splitlines() if line.strip()]
    assert all(len(r) == 7 for r in rows)
    return rows


def submit_stage(stage, policy, own_ids):
    job = returned_job(stage)
    if job:
        p = HERE/'intents'/(stage+'.submitted.json')
        if not p.exists():
            actual = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
            parse_allocation(actual, policy['cpus'], policy['memory_mib'])
            write(p, dict(job=job, stage=stage, actual_scontrol=actual, policy=policy))
        return job
    a = admission(queue(), own_ids, policy['cpus'], policy['memory_mib'])
    if not a['passed']:
        print('ADMISSION_WAIT', stage, a['used_cpus'], a['used_mib'], flush=True)
        return None
    args = ['sbatch', '--parsable', '--partition=mit_normal', '--account='+ACCOUNT, '--qos='+ACCOUNT,
            '--nodes=1', '--ntasks=1', '--cpus-per-task='+str(policy['cpus']),
            '--mem='+str(policy['memory_mib'])+'M', '--time='+str(policy['minutes']),
            '--exclude=node1622', '--comment='+token(stage), '--job-name=kr10f_'+stage,
            '--output='+str(HERE/'logs'/(stage+'_%j.out')), str(HERE/'qualified_job.sh'), str(HERE), stage]
    write(HERE/'intents'/(stage+'.intent.json'), dict(command=args, token=token(stage), admission=a))
    r = subprocess.run(args, text=True, capture_output=True, timeout=60, check=True)
    job = r.stdout.strip().split(';')[0]
    assert job.isdigit(), r.stdout
    write(HERE/'intents'/(stage+'.returned.json'), dict(job=job, token=token(stage), stdout=r.stdout))
    actual = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    parse_allocation(actual, policy['cpus'], policy['memory_mib'])
    write(HERE/'intents'/(stage+'.submitted.json'), dict(job=job, stage=stage, actual_scontrol=actual, policy=policy))
    print('SUBMITTED', stage, job, policy['cpus'], policy['memory_mib'], flush=True)
    return job


def tick():
    verify()
    with single_writer(HERE/'controller.lock'):
        plan = dag()
        jobs = {p.name.removesuffix('.returned.json'): read(p)['job'] for p in (HERE/'intents').glob('*.returned.json')}
        rows = queue()
        active = {r[0] for r in rows}
        outstanding = sum(j in active for j in jobs.values())
        failures = sorted(p.stem for p in (HERE/'failures').glob('*.json'))
        for stage, policy in plan.items():
            if (HERE/'receipts'/(stage+'.json')).exists():
                receipt(stage)
                continue
            if stage in jobs:
                if jobs[stage] not in active:
                    print('HOLD_MISSING_RECEIPT', stage, jobs[stage], flush=True)
                continue
            if outstanding >= 8:
                break
            if any(not (HERE/'receipts'/(dep+'.json')).exists() for dep in policy['deps']):
                continue
            for dep in policy['deps']:
                receipt(dep)
            if submit_stage(stage, policy, set(jobs.values())):
                outstanding += 1
        if failures:
            print('ISOLATED_FAILURES', failures, flush=True)
        return (HERE/'receipts/final.json').exists()


if __name__ == '__main__':
    from common import pins_ok
    guard=read(HERE/'QUALIFIED_LAUNCHER_LOCK.json')
    assert guard['base_lock_sha256']==sha(HERE/'LOCK.json')
    pins_ok({HERE/n:h for n,h in guard['files'].items()})
    verify()
    for name in ('logs', 'intents', 'receipts', 'failures'):
        (HERE/name).mkdir(exist_ok=True)
    deadline = time.monotonic() + (100*60 if len(sys.argv)>1 and sys.argv[1]=='watch' else 0)
    while True:
        if tick() or time.monotonic() >= deadline:
            break
        time.sleep(20)
