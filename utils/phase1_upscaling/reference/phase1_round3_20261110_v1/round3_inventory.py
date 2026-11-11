"""Compute-only, bounded, read-only audit of the terminal round-two solve scope."""
import collections
import datetime
import os
import traceback
import tomllib
import common as C
import recovery as R
import slurm as S
import worker


def audit():
    C.require('SLURM_JOB_ID' in os.environ, 'Inventory must run on compute')
    lock = C.verify_source(); R.parents_terminal(); out = R.INVENTORY
    C.require(not (out/'PASS.json').exists() and not (out/'FAILED.json').exists(), 'Inventory already attempted')
    queue = S.command(['squeue', '-h', '-r', '-u', os.environ['USER'], '-o', '%j'])
    C.require(not any(x.startswith(('p1q_num_', 'p1q_r2', 'p1q_r3s_', 'p1q_r3f_')) for x in queue.splitlines()), 'Prior/live numerical jobs present')
    for folder in (R.PREVIOUS/'solve/submissions').iterdir():
        if (folder/'INTENT.json').exists():
            C.require((folder/'ACK.json').exists() and (folder/'RELEASED.json').exists(), 'Unacknowledged or unreleased prior submission')
    parent = C.read(R.PREVIOUS_CODE/'LOCK.json')
    C.require(parent['science'] == lock['science'] and parent['method_sha256'] == lock['method_sha256'] and parent['configuration_sha256'] == lock['configuration_sha256'], 'Scientific identity changed')
    scope = C.read(R.PREVIOUS/'solve/SCOPE.json')
    counts = collections.Counter(); files = {}; selected = []; protected = {}; jobs = {}
    for case in scope['cases']:
        original_path = R.PRODUCTION/'plans'/(case+'.json')
        complete = R.PRODUCTION/'complete_cases'/(case+'.json')
        if complete.exists():
            d = C.read(complete)
            C.require(d['case_id'] == case and d['positions'] == 522 and d['method'] == lock['method_sha256'] and d['plan_sha256'] == C.sha(original_path), 'Completed-case marker changed')
            counts['complete_cases_skipped'] += 1
            continue
        previous_path = R.PREVIOUS/'solve/plans'/(case+'.json')
        plan = C.read(previous_path); seal = C.read(previous_path.with_suffix('.seal.json'))
        C.require(seal['sha256'] == C.sha(previous_path) and seal['source'] == scope['source'], 'Prior plan seal changed')
        C.require(C.sha(original_path) == plan['original_plan_sha256'], 'Original plan changed')
        original = {C.key(r): r for r in C.read(original_path)['records']}
        terminal_path = R.PREVIOUS/'solve/cases'/case/'TERMINAL.json'
        terminal = C.read(terminal_path); launch = C.read(terminal_path.with_name('LAUNCHED.json'))
        C.require(terminal['job'] == launch['job'] and len(terminal['tasks']) == len(plan['packets']) == launch['packets'], 'Prior array not fully reconciled')
        C.require(all(v[0] in ('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED') for v in terminal['tasks'].values()), 'Prior task not terminal')
        jobs[terminal['job']] = len(plan['packets']); rows = []
        for index, packet in enumerate(plan['packets']):
            for record in packet:
                key = C.key(record); slot = record['slot']; archive = R.PRODUCTION/'results'/case/f'core{slot:03d}.zip'
                C.require(original[key] == record and plan['entries'][key]['record_sha256'] == C.digest(record), 'Exact input differs')
                old = R.PREVIOUS/'solve/runs'/case/'cores'/key
                kind = R.original_state(old, archive)
                counts[kind] += 1
                if kind == 'archive_excluded':
                    C.verify_archive(archive, record, lock['method_sha256'])
                    continue
                claim = R.PREVIOUS/'numerical_claims'/(key+'.json')
                entry = dict(key=key, slot=slot, category=kind, record_sha256=C.digest(record),
                    original_work=str(old), original_job=terminal['job'], original_packet=index,
                    original_state=terminal['tasks'][str(index)], terminal_sha256=C.sha(terminal_path),
                    previous_claim_sha256=C.sha(claim) if claim.exists() else None)
                R.validate_prior(entry, record)
                for f in (old/'input/INPUT_PASS.toml', old/'PRIOR_INPUT_PASS.toml', R.PRODUCTION/'runs'/case/'cores'/key/'input/INPUT_PASS.toml'):
                    if f.exists():
                        entry.update(input_proof_path=str(f), input_proof_sha256=C.sha(f))
                        break
                if kind == 'interrupted':
                    C.require(entry.get('input_proof_path'), 'Interrupted input proof missing')
                    source = C.read(old/'input/SOURCE.json')
                    C.require(C.read(old/'input/record.json') == record and source['method'] == lock['method_sha256'] and source['configuration'] == lock['configuration_sha256'], 'Interrupted exact input identity changed')
                if kind == 'saved_flow':
                    worker.flow_proof(old, record, lock['method_sha256'])
                    entry['flow_proof_sha256'] = C.sha(old/'flow/CORE_PASS.toml')
                    for folder in ('input', 'flow'):
                        for f in sorted((old/folder).rglob('*')):
                            if f.is_file():
                                C.require(not f.is_symlink(), 'Saved checkpoint symlink')
                                protected[str(f)] = C.sha(f)
                if kind in ('saved_flow', 'interrupted', 'unstarted'):
                    rows.append(entry)
        if rows:
            dest = out/'cases'/(case+'.json')
            C.write(dest, dict(case_id=case, plan_sha256=C.sha(original_path), original_job=terminal['job'], cores=rows))
            files[case] = C.sha(dest); selected.append(case)
        print('AUDITED', case, dict(counts), flush=True)
    C.write(out/'SCOPE.json', dict(cases=selected, previous_scope_sha256=C.sha(R.PREVIOUS/'solve/SCOPE.json')))
    C.write(out/'PROTECTED_SAVED_FILES.json', protected)
    C.write(out/'PASS.json', dict(passed=True, source=C.sha(C.CODE/'LOCK.json'), counts=dict(counts),
        case_files=files, scope_sha256=C.sha(out/'SCOPE.json'), protected_saved_files_sha256=C.sha(out/'PROTECTED_SAVED_FILES.json'),
        previous_terminal_sha256=C.sha(R.PREVIOUS/'ROUND_TWO_TERMINAL.json'), previous_jobs=jobs,
        production_mutations=0, new_solves=0, job=os.environ['SLURM_JOB_ID'],
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('ROUND3_INVENTORY_PASS', dict(counts), flush=True)


if __name__ == '__main__':
    try:
        audit()
    except Exception:
        C.write(R.INVENTORY/'FAILED.json', dict(error=traceback.format_exc(), source=C.sha(C.CODE/'LOCK.json'), new_solves=0))
        raise
