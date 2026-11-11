"""Finite, two-stage recovery. Original assignments and first-pass work are immutable."""
import argparse
import collections
import contextlib
import datetime
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import common as C
import slurm as S
from control import terminal_tasks

BASE = Path('/orcd/data/juanes/001/shaowen')
PRODUCTION = BASE / 'phase1_pc40_adaptive10_campaign_20261030_v1'
HOME = PRODUCTION / 'round2_20261109_v1'
INVENTORY = BASE / 'phase1_round2_inventory_20261109_v2' / 'output'
INVENTORY_PIN = '06ad6ae069906992bf1f0257c85ba76c6aa0ba33d386f5f396dc3f5ced8515b4'
PARENT_PIN = 'c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf'
PHASES = ('finish', 'pilot', 'solve')
MAX_WORKERS = 128
MAX_CASES = 8


def production():
    return PRODUCTION


@contextlib.contextmanager
def at_root(root):
    prior = C.ROOT
    C.ROOT = root
    try:
        yield
    finally:
        C.ROOT = prior


def first_pass_terminal():
    C.require(C.read(PRODUCTION / 'STATE.json') == dict(active=[], cursor=2592), 'First pass changed or resumed')
    C.require(C.read(PRODUCTION / 'RUN.json')['source'] == PARENT_PIN, 'First-pass routing changed')
    C.require(C.read(PRODUCTION / 'FIRST_PASS_TERMINAL.json')['cases'] == 2592, 'First-pass terminal proof missing')


def authorize_recovery(scope):
    C.require(scope['purpose'] == 'round2' and scope['phase'] in PHASES, 'Invalid recovery phase')
    C.require(C.ROOT == HOME / scope['phase'], 'Wrong recovery output root')
    from window_contract import authorize_window
    authorize_window(scope, HOME)
    C.require(scope['production'] == str(PRODUCTION), 'Scope/production mismatch')
    if scope['phase'] != 'finish':
        C.require((HOME / 'finish/PHASE_TERMINAL.json').exists(), 'Saved-result stage must finish first')
    if scope['phase'] == 'solve':
        p = C.read(HOME / 'NUMERICAL_PILOT_PASS.json')
        from window_contract import PARENT_SOURCE
        C.require(p['passed'] and p['source'] == PARENT_SOURCE, 'Numerical lifecycle pilot not passed')
    first_pass_terminal()


def checked_plan(case):
    path = C.ROOT / 'plans' / (case + '.json')
    plan = C.read(path)
    seal = C.read(path.with_suffix('.seal.json'))
    C.require(seal['sha256'] == C.sha(path) and seal['source'] == C.read(C.ROOT / 'SCOPE.json')['source'], 'Recovery plan seal changed')
    original_path = PRODUCTION / 'plans' / (case + '.json')
    C.require(C.sha(original_path) == plan['original_plan_sha256'], 'Original fixed plan changed')
    original = {C.key(r): r for r in C.read(original_path)['records']}
    keys = []
    for packet in plan['packets']:
        C.require(0 < len(packet) <= 32, 'Invalid recovery packet')
        for r in packet:
            key = C.key(r)
            C.require(original.get(key) == r, 'Recovery input differs from original fixed assignment')
            C.require(plan['entries'][key]['record_sha256'] == C.digest(r), 'Recovery identity mismatch')
            keys.append(key)
    C.require(len(keys) == len(set(keys)) and set(keys) == set(plan['entries']), 'Duplicate or missing recovery assignment')
    return plan


def original_state(work, archive):
    if archive.exists():
        return 'archive_excluded'
    if (work / 'flow/CORE_PASS.toml').exists():
        return 'saved_flow'
    if any((work / 'flow' / n).exists() for n in ('pc100.csv', 'kr.csv', 'summary.toml')):
        return 'hold_unreceipted_output'
    if any((work / n).exists() for n in ('HOLD.json', 'FLOW_ERROR.toml', 'POSTFLOW_HOLD.json')):
        return 'hold_error'
    return 'interrupted' if (work / 'SOLVE_STARTED.toml').exists() else 'unstarted'


def copied_file(source, destination):
    C.require(source.is_file() and not source.is_symlink(), 'Unsafe saved file')
    h = C.sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with source.open('rb') as inp, destination.open('xb') as out:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
    C.require(C.sha(destination) == h, 'Saved checkpoint copy differs')
    return h


def prepare_work(plan, records, index):
    import worker
    phase = C.read(C.ROOT / 'SCOPE.json')['phase']
    method = C.read(C.CODE / 'LOCK.json')['method_sha256']
    for r in records:
        case = r['case_id']; key = C.key(r)
        archive = PRODUCTION / 'results' / case / f"core{r['slot']:03d}.zip"
        old = PRODUCTION / 'runs' / case / 'cores' / key
        new = C.ROOT / 'runs' / case / 'cores' / key
        if archive.exists():
            C.verify_archive(archive, r, method)
            continue
        entry = plan['entries'][key]
        kind = original_state(old, archive)
        C.require(kind == entry['category'], 'Original core changed; never infer permission to recompute')
        if phase == 'finish':
            C.require(kind == 'saved_flow', 'Finish-only scope contains numerical work')
            C.require(C.sha(old / 'flow/CORE_PASS.toml') == entry['flow_proof_sha256'], 'Saved flow receipt changed')
            worker.flow_proof(old, r, method)
            hashes = {}
            for folder in ('input', 'flow'):
                for f in sorted((old / folder).rglob('*')):
                    if f.is_file():
                        rel = f.relative_to(old)
                        hashes[rel.as_posix()] = copied_file(f, new / rel)
            C.write(new / 'RECOVERY.json', dict(mode='saved_flow_only', original_work=str(old), original_files=hashes,
                original_plan_sha256=plan['original_plan_sha256'], record_sha256=C.digest(r), new_solves=0))
            worker.flow_proof(new, r, method)
        else:
            C.require(kind in ('interrupted', 'unstarted'), 'Saved, accepted or ambiguous work cannot be recomputed')
            claim = dict(key=key, phase=phase, packet=index, source=C.sha(C.CODE / 'LOCK.json'), record_sha256=C.digest(r))
            # Immutable across pilot and production: a failed pilot is never retried by a new label.
            C.write(HOME / 'numerical_claims' / (key + '.json'), claim)
            new.mkdir(parents=True, exist_ok=True)
            if kind == 'interrupted':
                proof = old / 'input/INPUT_PASS.toml'
                C.require(C.sha(proof) == entry['input_proof_sha256'], 'Original interrupted input changed')
                copied_file(proof, new / 'PRIOR_INPUT_PASS.toml')
            C.write(new / 'RECOVERY.json', dict(mode=kind, original_work=str(old), original_job=plan['original_job'],
                original_plan_sha256=plan['original_plan_sha256'], record_sha256=C.digest(r), attempt=1,
                preserved_first_pass_outputs=True, completed_results_excluded=True))


def packetize(rows, records, phase):
    lookup = {r['slot']: r for r in records}
    packets = []
    if phase == 'finish':
        selected = [lookup[x['slot']] for x in rows]
        return [selected[i:i+32] for i in range(0, len(selected), 32)]
    interrupted = [lookup[x['slot']] for x in rows if x['category'] == 'interrupted']
    unstarted = [lookup[x['slot']] for x in rows if x['category'] == 'unstarted']
    packets.extend([[r] for r in interrupted])
    packets.extend([unstarted[i:i+8] for i in range(0, len(unstarted), 8)])
    return packets


def initialize(phase):
    C.require(C.ROOT == HOME / phase and phase in PHASES, 'Wrong initialization phase')
    existing = C.ROOT / 'SCOPE.json'
    if existing.exists():
        return C.read(existing)
    audit = C.read(INVENTORY / 'PASS.json')
    C.require(audit['passed'] and audit['source'] == INVENTORY_PIN and audit['cases'] == 229, 'Inventory not qualified/complete')
    cases = C.read(INVENTORY / 'SCOPE.json')['cases']
    selected_cases = []; selected_count = 0; pilot_kinds = set()
    exclusions = collections.Counter()
    for case in cases:
        inv = C.read(INVENTORY / 'cases' / (case + '.json'))
        path = PRODUCTION / 'plans' / (case + '.json')
        C.require(C.sha(path) == inv['plan_sha256'], 'Original plan changed after inventory')
        original = C.read(path)
        if (PRODUCTION / 'complete_cases' / (case + '.json')).exists():
            exclusions['full_case_completed'] += 1
            continue
        rows = []
        for entry in inv['cores']:
            slot = entry['slot']; key = entry['key']
            archive = PRODUCTION / 'results' / case / f'core{slot:03d}.zip'
            if archive.exists():
                exclusions['archive_excluded'] += 1
                continue
            kind = entry['category']
            permitted = kind == 'saved_flow' if phase == 'finish' else kind in ('interrupted', 'unstarted')
            if not permitted:
                exclusions[kind] += 1
                continue
            if phase != 'finish' and (HOME / 'numerical_claims' / (key + '.json')).exists():
                exclusions['prior_round_two_claim'] += 1
                continue
            if phase == 'pilot':
                if kind in pilot_kinds:
                    continue
                pilot_kinds.add(kind)
            rows.append(entry)
        if not rows:
            continue
        packets = packetize(rows, original['records'], phase)
        C.require(0 < len(packets) <= MAX_WORKERS, 'Case exceeds qualified worker window')
        plan = dict(case_id=case, purpose='round2', phase=phase, packets=packets,
            entries={r['key']: r for r in rows}, original_plan_sha256=inv['plan_sha256'], original_job=inv['original_job'])
        dest = C.ROOT / 'plans' / (case + '.json')
        C.write(dest, plan)
        C.write(dest.with_suffix('.seal.json'), dict(sha256=C.sha(dest), source=C.sha(C.CODE / 'LOCK.json')))
        selected_cases.append(case); selected_count += len(rows)
    if phase == 'pilot':
        C.require(pilot_kinds == {'interrupted', 'unstarted'}, 'Pilot must exercise interrupted and unstarted work')
        C.require(selected_count == 2, 'Pilot exceeded two new cores')
    scope = dict(purpose='round2', phase=phase, cases=selected_cases, source=C.sha(C.CODE / 'LOCK.json'),
        production=str(PRODUCTION), inventory_sha256=C.sha(INVENTORY / 'PASS.json'), cores=selected_count,
        exclusions=dict(exclusions), max_cases=MAX_CASES, max_worker_reservations=MAX_WORKERS)
    C.write(existing, scope)
    C.write(C.ROOT / 'STATE.json', dict(cursor=0, active=[]))
    return scope


def launch(case, scope):
    count = len(checked_plan(case)['packets'])
    from window_contract import preserved_launch
    if preserved_launch(case, count, HOME):
        return
    array = '0' if count == 1 else f'0-{count-1}'
    label = 'r2' + scope['phase'][0] + '_' + case
    args = [case] + (['--finish-only'] if scope['phase'] == 'finish' else [])
    w = S.spec(label, 'worker', args, array=array, cpus=2, memory=18432,
               minutes=30 if scope['phase'] == 'finish' else 120)
    job = S.held(w)
    g = S.spec(label + '_gate', 'recovery', ['gate', case], dependency=job, minutes=15)
    gate = S.held(g)
    C.write(C.ROOT / 'cases' / case / 'LAUNCHED.json', dict(job=job, gate=gate, packets=count))
    S.release(gate, g); S.release(job, w)


def close_case(case):
    dest = PRODUCTION / 'complete_cases' / (case + '.json')
    if dest.exists():
        return
    path = PRODUCTION / 'plans' / (case + '.json')
    original = C.read(path)
    paths = [PRODUCTION / 'runs' / case / 'done' / f"core{r['slot']:03d}.json" for r in original['records']]
    if not all(p.exists() for p in paths):
        return
    method = C.read(C.CODE / 'LOCK.json')['method_sha256']
    for r, receipt_path in zip(original['records'], paths):
        receipt = C.read(receipt_path)
        C.require(receipt['key'] == C.key(r) and receipt['record_sha256'] == C.digest(r), 'Case receipt identity')
        C.require(receipt['scientific_pass'] and receipt['method'] == method, 'Case scientific proof')
        archive = PRODUCTION / 'results' / case / f"core{r['slot']:03d}.zip"
        C.require(archive.is_file() and archive.stat().st_size == receipt['bytes'], 'Case archive missing/truncated')
    C.write(dest, dict(case_id=case, positions=522, plan_sha256=C.sha(path), method=method))


def pump():
    C.verify_source(); scope = C.read(C.ROOT / 'SCOPE.json'); authorize_recovery(scope)
    with C.locked(C.ROOT / 'control.lock'):
        state = C.read(C.ROOT / 'STATE.json')
        state['active'] = [c for c in state['active'] if not (C.ROOT / 'cases' / c / 'TERMINAL.json').exists()]
        C.write(C.ROOT / 'STATE.json', state, replace=True)
        for c in state['active']:
            launch(c, scope)
        # Sparse recovery cases may share the unchanged 128-packet budget.
        case_limit = MAX_WORKERS if scope['phase'] == 'solve' else MAX_CASES
        while state['cursor'] < len(scope['cases']) and len(state['active']) < case_limit:
            case = scope['cases'][state['cursor']]
            occupied = sum(len(checked_plan(c)['packets']) for c in state['active'])
            required = len(checked_plan(case)['packets'])
            if occupied + required > MAX_WORKERS:
                break
            # Count unrelated submitted tasks without recognizing or controlling their workflow.
            q = S.command(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%i'])
            C.require(len(q.splitlines()) + required + 1 <= 440, 'Submitted-job capacity unavailable; existing assignments retained')
            state['active'].append(case); state['cursor'] += 1
            C.write(C.ROOT / 'STATE.json', state, replace=True)
            launch(case, scope)
        exhausted = state['cursor'] == len(scope['cases']) and not state['active']
        if exhausted:
            C.write(C.ROOT / 'PHASE_TERMINAL.json', dict(phase=scope['phase'], cases=len(scope['cases']),
                planned_cores=scope['cores'], source=scope['source'], scientific_completion=False))
    if exhausted:
        next_phase(scope['phase'])


def next_phase(phase):
    if phase == 'solve':
        C.write(HOME / 'ROUND_TWO_TERMINAL.json', dict(source=C.sha(C.CODE / 'LOCK.json'),
            scientific_completion=False, note='Review remaining exceptions; no blind retries'))
        return
    with C.locked(HOME / 'phase_transition.lock'):
        if phase == 'pilot':
            scope = C.read(C.ROOT / 'SCOPE.json')
            method = C.read(C.CODE / 'LOCK.json')['method_sha256']
            proofs = []
            for case in scope['cases']:
                for packet in checked_plan(case)['packets']:
                    for r in packet:
                        target = PRODUCTION / 'results' / case / f"core{r['slot']:03d}.zip"
                        C.require(target.exists(), 'Numerical pilot did not finish: remaining numerical launch stopped')
                        C.verify_archive(target, r, method)
                        proofs.append(dict(key=C.key(r), archive_sha256=C.sha(target)))
            C.require(len(proofs) == 2, 'Numerical pilot coverage')
            C.write(HOME / 'NUMERICAL_PILOT_PASS.json', dict(passed=True, source=C.sha(C.CODE / 'LOCK.json'),
                accepted_new_cores=proofs, completed_first_pass_results_recomputed=0))
        following = 'pilot' if phase == 'finish' else 'solve'
        with at_root(HOME / following):
            initialize(following)
            # Arm one finite bootstrap; no resident manager or second seed.
            w = S.spec('r2_' + following + '_bootstrap', 'recovery', ['pump'], cpus=1, memory=1024, minutes=45)
            job = S.held(w); S.release(job, w)


def gate(case):
    scope = C.read(C.ROOT / 'SCOPE.json'); authorize_recovery(scope)
    launch_record = C.read(C.ROOT / 'cases' / case / 'LAUNCHED.json')
    for attempt in range(12):
        raw = S.command(['sacct', '-X', '-n', '-P', '-j', launch_record['job'], '--format=JobID%40,State,ExitCode'])
        try:
            states = terminal_tasks(raw, launch_record['job'], launch_record['packets'])
            break
        except RuntimeError:
            if attempt == 11:
                raise
            time.sleep(5)
    close_case(case)
    C.write(C.ROOT / 'cases' / case / 'TERMINAL.json', dict(job=launch_record['job'], tasks=states))
    pump()


def start():
    C.require(C.ROOT == HOME, 'Wrong round-two home')
    first_pass_terminal()
    q = S.command(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%j'])
    C.require(not any(x.startswith('p1q_num_') for x in q.splitlines()), 'First-pass jobs still present')
    with C.locked(HOME / 'phase_transition.lock'):
        with at_root(HOME / 'finish'):
            initialize('finish')
            authorize_recovery(C.read(C.ROOT / 'SCOPE.json'))
    with at_root(HOME / 'finish'):
        pump()


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['start', 'pump', 'gate']); p.add_argument('case', nargs='?')
    a = p.parse_args()
    C.require('SLURM_JOB_ID' in os.environ, 'Recovery orchestration runs on compute')
    if a.mode == 'start':
        start()
    elif a.mode == 'pump':
        pump()
    else:
        gate(a.case)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        C.write(C.ROOT / 'errors' / (os.environ.get('SLURM_JOB_ID', 'operator') + '.json'),
                dict(error=traceback.format_exc(), automatic_retry=False))
        raise
