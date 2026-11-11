"""One-time checked cutover and finite qualification jobs, never a second pump."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import time
import unittest
import common as C
import slurm as S

PRODUCTION = Path('/orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_campaign_20261030_v1')

@contextlib.contextmanager
def location(root, code=None):
    before = C.ROOT, C.CODE
    C.ROOT = Path(root)
    if code is not None:
        C.CODE = Path(code)
    try:
        yield
    finally:
        C.ROOT, C.CODE = before

def queue():
    raw = S.command(['squeue', '-u', 'shaowen', '-r', '-h', '-o', '%i|%T|%r'])
    return {v[0]: dict(state=v[1], reason=v[2]) for line in raw.splitlines() if (v := line.split('|')) and len(v) == 3}

def submissions():
    paths = sorted((C.ROOT / 'submissions').glob('*/INTENT.json'))
    C.require(len(paths) <= 6000, 'Unexpected submission inventory size')
    result = {}
    for p in paths:
        intent = C.read(p)
        ack = C.read(p.with_name('ACK.json'))
        C.require(intent['spec'] == ack['spec'], 'Unreconciled submission: ' + p.parent.name)
        C.require(ack['job'] not in result, 'Duplicate job assignment')
        result[ack['job']] = ack
    return result

def hold_handoffs():
    """Hold pending gates only; allow any running gate to finish before cutover."""
    journal = C.ROOT / 'cache_transition_gate_holds'
    for attempt in range(24):
        jobs = submissions()
        live = queue()
        running = []
        for job, ack in jobs.items():
            if ack['spec']['mode'] != 'control' or job not in live:
                continue
            state = live[job]
            if state['state'] in ('RUNNING', 'COMPLETING'):
                running.append(job)
                continue
            C.require(state['state'] == 'PENDING', 'Unexpected handoff state')
            note = journal / (job + '.json')
            if state['reason'] == 'JobHeldUser':
                C.require(note.exists(), 'Preexisting user-held handoff; not ours to release')
                continue
            C.require(state['reason'] not in ('JobHeldAdmin', 'DependencyNeverSatisfied'), 'Blocked handoff')
            if note.exists():
                C.require(C.read(note)['job'] == job and C.read(note)['spec'] == ack['spec'], 'Hold intent changed')
            else:
                C.write(note, dict(job=job, spec=ack['spec'], before=state))
            # A start between observation and hold is safe: reobserve, never cancel.
            try:
                S.command(['scontrol', 'hold', job])
            except subprocess.CalledProcessError:
                current = queue().get(job)
                C.require(current is None or current['state'] in ('RUNNING', 'COMPLETING'), 'Handoff hold failed')
                running.append(job)
        now = queue()
        gates = [j for j, a in submissions().items() if a['spec']['mode'] == 'control' and j in now]
        if not running and gates and all(now[j]['state'] == 'PENDING' and now[j]['reason'] == 'JobHeldUser' for j in gates):
            return
        time.sleep(5)
    raise RuntimeError('No safe handoff boundary within two minutes; no cutover')

def release_handoffs():
    live = queue()
    for p in sorted((C.ROOT / 'cache_transition_gate_holds').glob('*.json')):
        job = C.read(p)['job']
        if job in live and live[job]['state'] == 'PENDING' and live[job]['reason'] == 'JobHeldUser':
            S.command(['scontrol', 'release', job])

def snapshot(fixture=False):
    lock = C.verify_source()
    old_run = dict(code=lock['legacy_code'], source=lock['legacy_source'])
    C.require(C.read(C.ROOT / 'RUN.json') == old_run, 'Production runner already changed')
    C.require(C.sha(Path(old_run['code']) / 'LOCK.json') == old_run['source'], 'Old source identity')
    scope = C.read(C.ROOT / 'SCOPE.json')
    C.require(scope['purpose'] == ('lifecycle_fixture' if fixture else 'production'), 'Wrong transition scope')
    if not fixture:
        C.require(C.ROOT == PRODUCTION and len(scope['cases']) == 2592 and scope['packet_count'] == 17, 'Production scope mismatch')
        auth = C.read(C.ROOT / 'LAUNCH_AUTHORIZATION.json')
        C.require(auth['approved_by_user'] and auth['source_sha256'] == old_run['source'] and auth['scope_sha256'] == C.digest(scope), 'Original authorization changed')
        C.require(scope['configuration_sha256'] == lock['configuration_sha256'], 'Frozen configuration changed')
    state = C.read(C.ROOT / 'STATE.json')
    C.require(len(state['active']) == len(set(state['active'])) <= 8 and state['active'], 'Invalid active window')
    C.require(all(c in scope['cases'][:state['cursor']] for c in state['active']), 'Invalid frontier')
    jobs = submissions()
    live = queue()
    C.require(all(v['state'] == 'PENDING' and v['reason'] == 'JobHeldUser' for j, v in live.items() if j in jobs and jobs[j]['spec']['mode'] == 'control'), 'A legacy handoff is not quiescent')
    rows = {}
    for case in state['active']:
        launch_path = Path('cases') / case / 'LAUNCHED.json'
        launch = C.read(C.ROOT / launch_path)
        files = {str(launch_path): C.sha(C.ROOT / launch_path)}
        for role, job in launch.items():
            ack = jobs[job]
            want = ack['spec']
            C.require(want['root'] == str(C.ROOT) and want['source'] == old_run['source'] and want['script'] == old_run['code'] + '/job.sh', 'Legacy submission routing mismatch')
            C.require(want['args'] == (['gate', case] if role == 'gate' else [case]), 'Legacy case assignment mismatch')
            C.require(want['mode'] == ('control' if role == 'gate' else ('fixture' if fixture else 'worker')), 'Legacy submission mode')
            if role == 'job':
                C.require(want['array'] == ('0-1' if fixture else '0-16'), 'Legacy array size')
            folder = Path('submissions') / want['label']
            for name in ('INTENT.json', 'ACK.json', 'ALLOCATION.json', 'RELEASED.json'):
                p = C.ROOT / folder / name
                if fixture and not p.exists():
                    continue
                files[str(folder / name)] = C.sha(p)
            if not fixture:
                C.require(C.read(C.ROOT / folder / 'RELEASED.json') == dict(job=job, spec_sha256=C.digest(want)), 'Legacy release identity')
            if role == 'gate':
                C.require(job in live and live[job]['reason'] == 'JobHeldUser', 'Active successor missing or unheld')
        if not fixture:
            plan = Path('plans') / (case + '.json')
            seal = plan.with_suffix('.seal.json')
            proof = C.read(C.ROOT / seal)
            C.require(proof == dict(source=old_run['source'], plan_sha256=C.sha(C.ROOT / plan)), 'Legacy plan seal')
            for p in (plan, seal):
                files[str(p)] = C.sha(C.ROOT / p)
        rows[case] = dict(launch=launch, files=files)
    active_jobs = {j for r in rows.values() for j in r['launch'].values()}
    C.require(all(j.split('_')[0] in active_jobs for j in live if j.split('_')[0] in jobs), 'Live job outside active assignments')
    return dict(approved_by_user=True, old_run=old_run,
                new_run=dict(code=str(C.CODE), source=C.sha(C.CODE / 'LOCK.json')),
                scope_sha256=C.sha(C.ROOT / 'SCOPE.json'),
                authorization_sha256=None if fixture else C.sha(C.ROOT / 'LAUNCH_AUTHORIZATION.json'),
                method=lock['method_sha256'], configuration=lock['configuration_sha256'],
                boundary_state=state, legacy_cases=rows, fixture=fixture,
                authorization='Future jobs use qualified cache; existing numerical jobs/results unchanged; rolling eight cases retained')

def activate(target, fixture=False):
    with location(target):
        if not fixture:
            C.require(C.read(C.CODE / 'qualification/UNIT_PASS.json')['source'] == C.sha(C.CODE / 'LOCK.json'), 'Compute tests not qualified')
            q = C.read(C.CODE / 'qualification/FIXTURE_PASS.json')
            C.require(q['passed'] and q['source'] == C.sha(C.CODE / 'LOCK.json'), 'Slurm transition lifecycle not qualified')
            passed = C.read(C.CODE.parent / 'phase1_pc40_adaptive10_runner_20261030_v4/numerical/PACKET_QUALIFIED_PASS.json')
            C.require(passed['passed'] and passed['source'] == C.read(C.CODE / 'LOCK.json')['predecessor'] and passed['verified_cores'] == 32, 'Packet qualification absent')
        try:
            if not fixture:
                hold_handoffs()
            with C.locked(C.ROOT / 'control.lock'):
                if (C.ROOT / 'CACHE_ADOPTION.json').exists():
                    from adoption import record
                    a = record()
                    current = C.read(C.ROOT / 'RUN.json')
                    C.require(current in (a['old_run'], a['new_run']), 'Uncertain routing state')
                    if current == a['old_run']:
                        C.require(snapshot(fixture) == a, 'Interrupted transition boundary changed')
                else:
                    a = snapshot(fixture)
                    C.write(C.ROOT / 'CACHE_ADOPTION.json', a)
                C.write(C.ROOT / 'RUN.json', a['new_run'], replace=True)
                applied = C.ROOT / 'CACHE_ADOPTION_APPLIED.json'
                if applied.exists():
                    receipt = C.read(applied)
                    C.require(receipt['source'] == a['new_run']['source'] and receipt['boundary_state'] == a['boundary_state'] and receipt['scientific_results_modified'] is False, 'Applied transition changed')
                else:
                    C.write(applied, dict(source=a['new_run']['source'], boundary_state=a['boundary_state'], job=os.environ.get('SLURM_JOB_ID'), scientific_results_modified=False))
            print('CUTOVER_APPLIED', str(C.ROOT), a['boundary_state'], flush=True)
        finally:
            if not fixture:
                a = C.read(C.ROOT / 'CACHE_ADOPTION.json') if (C.ROOT / 'CACHE_ADOPTION.json').exists() else None
                if a is None or C.read(C.ROOT / 'RUN.json') == a['new_run']:
                    release_handoffs()

def submit(label, args, dependency=None):
    (C.ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    want = S.spec(label, 'transition', args, dependency=dependency, cpus=1, memory=1024, minutes=10)
    job = S.held(want)
    S.release(job, want)
    print(label, job, flush=True)
    return job

def units():
    C.verify_source()
    suite = unittest.defaultTestLoader.loadTestsFromNames(['test_runner', 'test_cache_integration', 'test_transition'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    C.require(result.wasSuccessful(), 'Transition unit tests failed')
    C.write(C.CODE / 'qualification/UNIT_PASS.json', dict(passed=True, source=C.sha(C.CODE / 'LOCK.json'), tests=result.testsRun))

def fixture_init():
    root = C.CODE / 'fixture'
    C.require(not root.exists(), 'Fixture already exists; reconcile instead of recreating')
    old = Path(C.read(C.CODE / 'LOCK.json')['legacy_code'])
    with location(root, old):
        (root / 'logs').mkdir(parents=True)
        cases = [f'fixture{i:02d}' for i in range(9)]
        C.write(root / 'RUN.json', dict(code=str(old), source=C.sha(old / 'LOCK.json')))
        C.write(root / 'SCOPE.json', dict(purpose='lifecycle_fixture', cases=cases, packet_count=2))
        C.write(root / 'STATE.json', dict(cursor=8, active=cases[:8]))
        for case in cases[:8]:
            w = S.spec('life_' + case, 'fixture', [case], array='0-1', cpus=1, memory=256, minutes=5)
            job = S.held(w)
            g = S.spec(w['label'] + '_gate', 'control', ['gate', case], dependency=job)
            gate = S.held(g)
            C.write(root / 'cases' / case / 'LAUNCHED.json', dict(job=job, gate=gate))
            if case != 'fixture00':
                S.release(job, w)
        print('FIXTURE_ORIGINAL_JOBS_READY', flush=True)

def fixture_switch():
    root = C.CODE / 'fixture'
    activate(root, fixture=True)
    old = Path(C.read(C.CODE / 'LOCK.json')['legacy_code'])
    with location(root, old):
        # Release the deliberately queued old worker only after the routing switch.
        for case in [f'fixture{i:02d}' for i in range(8)]:
            folder = root / 'submissions' / ('life_' + case)
            ack = C.read(folder / 'ACK.json')
            S.release(ack['job'], ack['spec'])
            gate = C.read(folder.with_name(folder.name + '_gate') / 'ACK.json')
            S.release(gate['job'], gate['spec'])

def fixture_audit():
    root = C.CODE / 'fixture'
    with location(root):
        C.require(C.read(root / 'STATE.json') == dict(cursor=9, active=[]), 'Fixture window did not drain')
        C.require(C.read(root / 'FIRST_PASS_TERMINAL.json')['scientific_completion'] is False, 'Fixture credited as science')
        a = C.read(root / 'CACHE_ADOPTION.json')
        C.require(len(a['legacy_cases']) == 8, 'Fixture legacy window missing')
        from adoption import legacy_assignment
        for case in a['legacy_cases']:
            legacy_assignment(case)
        for i in range(9):
            case = f'fixture{i:02d}'
            terminal = C.read(root / 'cases' / case / 'TERMINAL.json')
            C.require(len(terminal['tasks']) == 2, 'Fixture terminal tasks')
            for task in range(2):
                expected = ['FAILED', '7:0'] if (i, task) == (2, 1) else ['COMPLETED', '0:0']
                C.require(terminal['tasks'][str(task)] == expected, 'Unexpected fixture task failure')
                out = C.read(root / 'fixture_outputs' / case / (str(task) + '.json'))
                C.require(out['fixture'] and not out['scientific_pass'], 'Fixture output scope')
            want = C.read(root / 'submissions' / ('life_' + case) / 'ACK.json')['spec']
            C.require(want['source'] == (C.sha(C.CODE / 'LOCK.json') if i == 8 else a['old_run']['source']), 'Wrong fixture runner')
        log = root / 'logs' / ('p1q_life_fixture00_' + a['legacy_cases']['fixture00']['launch']['job'] + '_0.log')
        C.require('PRESERVED_LEGACY_RUNNER' in log.read_text(), 'Queued legacy worker was not routed')
        C.require(not list((root / 'complete_cases').glob('*.json')), 'Fixture created scientific completion')
        C.write(C.CODE / 'qualification/FIXTURE_PASS.json', dict(passed=True, source=C.sha(C.CODE / 'LOCK.json'), cases=9, tasks=18,
            queued_legacy_preserved=True, failure_did_not_block=True, rolling_window=8, scientific_results_modified=False))
        print('FIXTURE_PASS', flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['units', 'submit-unit', 'init-fixture', 'fixture-switch', 'submit-fixture-switch', 'fixture-audit', 'submit-fixture-audit', 'submit-production-switch', 'apply'])
    p.add_argument('target', nargs='?')
    a = p.parse_args()
    C.verify_source()
    if a.mode == 'units':
        units()
    elif a.mode == 'init-fixture':
        fixture_init()
    elif a.mode == 'fixture-switch':
        fixture_switch()
    elif a.mode == 'fixture-audit':
        fixture_audit()
    elif a.mode == 'apply':
        C.require(Path(a.target) == PRODUCTION, 'Only authorized production root')
        activate(PRODUCTION)
    else:
        actions = {'submit-unit': ('cachev5_units', ['units']), 'submit-fixture-switch': ('cachev5_fixture_switch', ['fixture-switch']),
                   'submit-fixture-audit': ('cachev5_fixture_audit', ['fixture-audit']),
                   'submit-production-switch': ('cachev5_production_switch', ['apply', str(PRODUCTION)])}
        label, args = actions[a.mode]
        with location(C.CODE / 'operations'):
            submit(label, args)

if __name__ == '__main__':
    main()
