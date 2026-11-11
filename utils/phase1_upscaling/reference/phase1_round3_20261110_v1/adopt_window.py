"""One finite, compute-side replacement of pending recovery control gates only."""
import os
from pathlib import Path
import time
import traceback
import common as C
import recovery as R
import slurm as S
from transition_contract import require_held_gate, retired_gate, validate_authority
from window_contract import PARENT_SOURCE, PARENT_CODE, ADOPTION, TRANSITION


def fields(job):
    raw = S.command(['scontrol', 'show', 'job', '-o', job])
    rows = raw.splitlines()
    C.require(len(rows) == 1 and rows[0].startswith('JobId='+job+' '), 'Ambiguous control job')
    return dict(x.split('=', 1) for x in rows[0].split() if '=' in x)


def verify_held(job, want):
    actual = S.check(fields(job), want)
    C.require(actual['JobState'] == 'PENDING' and actual['Reason'] == 'JobHeldUser', 'Replacement not held')
    spool = S.command(['scontrol', 'write', 'batch_script', job, '-'])+'\n'
    C.require(spool == (C.CODE/'job.sh').read_text(), 'Replacement launcher mismatch')
    C.write(C.ROOT/'submissions'/want['label']/'ALLOCATION.json', actual)


def retire_gates(gates):
    # Never retry the enclosing transition or a numerical/submission body.
    for item in gates.values():
        job = item['old_gate']
        require_held_gate(fields(job), job, item['worker'], str(PARENT_CODE/'job.sh'))
        S.command(['scancel', job])
        for attempt in range(12):
            raw = S.command(['sacct', '-X', '-n', '-P', '-j', job, '--format=JobID%40,State,ExitCode'])
            try:
                retired_gate(raw, job)
                break
            except RuntimeError:
                if attempt == 11:
                    raise
                time.sleep(5)


def main():
    C.require('SLURM_JOB_ID' in os.environ, 'Transition must run on compute')
    out = R.HOME/'transitions'/TRANSITION
    C.require(C.ROOT == out, 'Wrong window-transition output')
    C.require(not (out/'BEGIN.json').exists(), 'Transition already attempted; reconcile, never rerun')
    C.write(out/'BEGIN.json', dict(job=os.environ['SLURM_JOB_ID'], source=C.sha(C.CODE/'LOCK.json')))
    lock = C.verify_source(); source = C.sha(C.CODE/'LOCK.json')
    q = C.read(C.CODE/'qualification/QUALIFIED_PASS.json')
    C.require(q['passed'] and q['source'] == source, 'Window runner not qualified')
    C.require(C.sha(PARENT_CODE/'LOCK.json') == PARENT_SOURCE, 'Parent source changed')
    parent = C.read(PARENT_CODE/'LOCK.json')
    for name, digest in parent['code'].items():
        C.require(C.sha(PARENT_CODE/name) == digest, 'Parent code changed')
    C.require(lock['science'] == parent['science'], 'Scientific implementation changed')
    C.require(not (R.HOME/ADOPTION).exists(), 'Window adoption already exists')
    phase = R.HOME/'solve'
    with C.locked(R.HOME/'phase_transition.lock'), C.locked(phase/'control.lock'):
        R.first_pass_terminal()
        scope = C.read(phase/'SCOPE.json'); state = C.read(phase/'STATE.json')
        validate_authority(scope, C.read(R.HOME/'AUTHORIZATION.json'),
                           C.read(R.HOME/'ADOPTION_budget_guard_20261109_v1.json'),
                           C.read(PARENT_CODE/'qualification/QUALIFIED_PASS.json'), PARENT_SOURCE, parent)
        C.require(scope['phase'] == 'solve' and scope['max_cases'] == 8 and scope['max_worker_reservations'] == 128,
                  'Not the original recovery window')
        C.require(0 < len(state['active']) <= 8 and len(set(state['active'])) == len(state['active']), 'Unexpected active window')
        C.require(set(state['active']).issubset(scope['cases'][:state['cursor']]), 'Unassigned active case')
        C.require(not list((phase/'errors').glob('*.json')), 'Reconcile existing control errors first')
        C.require(not (phase/'PHASE_TERMINAL.json').exists(), 'Recovery already terminal')
        # Every original reservation must already have its immutable ACK/release.
        for folder in (phase/'submissions').iterdir():
            if not folder.is_dir():
                continue
            intent = C.read(folder/'INTENT.json'); ack = C.read(folder/'ACK.json')
            C.require(ack['spec'] == intent['spec'] and ack['spec']['source'] == PARENT_SOURCE,
                      'Unreconciled or foreign submission')
            C.require(C.read(folder/'RELEASED.json') == dict(job=ack['job'], spec_sha256=C.digest(ack['spec'])),
                      'Original release uncertain')
        protected = [R.HOME/'AUTHORIZATION.json', R.HOME/'ADOPTION_budget_guard_20261109_v1.json',
                     R.HOME/'NUMERICAL_PILOT_PASS.json', phase/'SCOPE.json', phase/'STATE.json',
                     R.PRODUCTION/'STATE.json', R.PRODUCTION/'RUN.json', PARENT_CODE/'LOCK.json']
        gates = {}; occupied = 0
        with R.at_root(phase):
            for case in state['active']:
                C.require(not (phase/'cases'/case/'TERMINAL.json').exists(), 'Gate is already completing; retry only after reconciliation')
                plan = R.checked_plan(case); count = len(plan['packets']); occupied += count
                launch_path = phase/'cases'/case/'LAUNCHED.json'; launch = C.read(launch_path)
                C.require(launch['packets'] == count, 'Packet assignment changed')
                label = 'r2s_'+case
                w = C.read(phase/'submissions'/label/'ACK.json')
                g = C.read(phase/'submissions'/(label+'_gate')/'ACK.json')
                expected_w = S.spec(label, 'worker', [case], array='0' if count == 1 else f'0-{count-1}', cpus=2, memory=18432, minutes=120)
                expected_g = S.spec(label+'_gate', 'recovery', ['gate',case], dependency=launch['job'], minutes=15)
                for spec in (expected_w, expected_g):
                    spec.update(source=PARENT_SOURCE, script=str(PARENT_CODE/'job.sh'))
                C.require(w == dict(job=launch['job'], spec=expected_w) and g == dict(job=launch['gate'], spec=expected_g),
                          'Original worker/gate identity mismatch')
                f = fields(launch['gate'])
                C.require(f['JobState'] == 'PENDING' and f['Reason'] not in ('JobHeldAdmin','DependencyNeverSatisfied'),
                          'Control job is running or unsafe; do not cancel')
                S.check(f, expected_g)
                gates[case] = dict(worker=launch['job'], old_gate=launch['gate'])
                protected += [launch_path, phase/'plans'/(case+'.json'), phase/'plans'/(case+'.seal.json'),
                              phase/'submissions'/label/'ACK.json', phase/'submissions'/(label+'_gate')/'ACK.json']
                for packet in plan['packets']:
                    for r in packet:
                        claim = R.HOME/'numerical_claims'/(C.key(r)+'.json')
                        if claim.exists():
                            protected.append(claim)
                        archive = R.PRODUCTION/'results'/case/f"core{r['slot']:03d}.zip"
                        if archive.exists():
                            C.verify_archive(archive, r, lock['method_sha256']); protected.append(archive)
            C.require(occupied <= 128, 'Existing reservations exceed 128')
            queue = S.command(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j'])
            controls = {line.split('|')[0] for line in queue.splitlines()
                        if (line.split('|')[1].startswith('p1q_r2s_') and line.split('|')[1].endswith('_gate'))
                        or line.split('|')[1] == 'p1q_r2_solve_bootstrap'}
            C.require(controls == {x['old_gate'] for x in gates.values()}, 'Unexpected live recovery control; reconcile first')
            C.require(len(queue.splitlines())+len(gates)+1 <= 440, 'Insufficient submitted-job headroom for handoff replacement')
            before = {str(p):C.sha(p) for p in protected}
            C.write(out/'PLAN.json', dict(state=state, gates=gates, worker_reservations=occupied,
                                         source=source, protected_files=before))
            for item in gates.values():
                S.command(['scontrol','hold',item['old_gate']])
                require_held_gate(fields(item['old_gate']), item['old_gate'], item['worker'], str(PARENT_CODE/'job.sh'))
            C.write(out/'HELD.json', dict(old_gates=[x['old_gate'] for x in gates.values()]))
            replacements = []
            for case,item in gates.items():
                want = S.spec('r2s_'+case+'_window_v4_gate', 'recovery', ['gate',case], dependency=item['worker'], minutes=15)
                job = S.held(want); verify_held(job,want); replacements.append((job,want)); item['new_gate'] = job
            want = S.spec('r2_solve_window_v4_bootstrap', 'recovery', ['pump'],
                          dependency=os.environ['SLURM_JOB_ID'], minutes=45)
            bootstrap = S.held(want); verify_held(bootstrap,want); replacements.append((bootstrap,want))
            C.write(R.HOME/ADOPTION, dict(approved_by_user=True, previous_source=PARENT_SOURCE, source=source,
                method=lock['method_sha256'], configuration=lock['configuration_sha256'],
                scope_sha256=C.sha(phase/'SCOPE.json'), effective_max_cases=128, max_worker_reservations=128,
                counting='All original packets charged until whole case terminal; unchanged',
                gates=gates, bootstrap=bootstrap, qualification_sha256=C.sha(C.CODE/'qualification/QUALIFIED_PASS.json')))
            retire_gates(gates)
            for name,digest in before.items():
                C.require(C.sha(Path(name)) == digest, 'Protected result or authority changed')
            # Old controls are retired before any successor may submit new work.
            C.write(out/'TRANSITION_PASS.json', dict(passed=True, source=source, gates=gates, bootstrap=bootstrap,
                original_worker_arrays_untouched=[x['worker'] for x in gates.values()],
                completed_calculations_repeated=0, numerical_jobs_submitted_by_transition=0,
                protected_files=before, existing_worker_reservations=occupied, max_worker_reservations=128))
            for job,want in replacements:
                S.release(job,want)
            C.write(out/'RELEASES_PASS.json', dict(passed=True, source=source, controls=[x[0] for x in replacements]))
            print('RECOVERY_WINDOW_ADOPTED',bootstrap,len(gates),occupied,flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        C.write(C.ROOT/'FAILED.json',dict(job=os.environ.get('SLURM_JOB_ID'),error=traceback.format_exc(),automatic_retry=False))
        raise
