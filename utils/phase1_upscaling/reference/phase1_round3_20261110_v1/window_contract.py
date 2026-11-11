"""Recovery-only successor; never replace sealed plans or existing worker jobs."""
from pathlib import Path
import common as C
import slurm as S
from transition_contract import validate_authority

PARENT_SOURCE = 'facf4d6fb3f93f144c19dfd6862e7cf45c24803b6333b4d69d6031304adc2c6a'
PARENT_CODE = Path('/orcd/data/juanes/001/shaowen/phase1_round2_20261109_v3')
ADOPTION = 'ADOPTION_recovery_window_20261109_v1.json'
TRANSITION = 'recovery_window_20261109_v1'


def validate_window(scope, adoption, qualification, transition, source, lock, scope_hash):
    C.require(scope['phase'] == 'solve' and scope['source'] == PARENT_SOURCE,
              'Only the existing numerical recovery scope may expand')
    C.require(scope['max_cases'] == 8 and scope['max_worker_reservations'] == 128,
              'Original recovery limits changed')
    C.require(adoption['approved_by_user'] and adoption['previous_source'] == PARENT_SOURCE
              and adoption['source'] == source, 'Invalid recovery-window authority')
    C.require(adoption['scope_sha256'] == scope_hash, 'Sealed recovery scope changed')
    C.require(adoption['max_worker_reservations'] == 128 and adoption['effective_max_cases'] == 128,
              'Recovery ceiling changed')
    C.require(qualification['passed'] and qualification['source'] == source,
              'Window successor not qualified')
    C.require(transition['passed'] and transition['source'] == source,
              'Control transition not complete')
    for field in ('method', 'configuration'):
        C.require(adoption[field] == qualification[field] == lock[field+'_sha256'],
                  'Scientific identity changed')


def authorize_window(scope, home):
    C.require(C.sha(PARENT_CODE/'LOCK.json') == PARENT_SOURCE, 'Predecessor source changed')
    parent_lock = C.read(PARENT_CODE/'LOCK.json')
    parent_q = C.read(PARENT_CODE/'qualification/QUALIFIED_PASS.json')
    validate_authority(scope, C.read(home/'AUTHORIZATION.json'),
                       C.read(home/'ADOPTION_budget_guard_20261109_v1.json'),
                       parent_q, PARENT_SOURCE, parent_lock)
    validate_window(scope, C.read(home/ADOPTION), C.read(C.CODE/'qualification/QUALIFIED_PASS.json'),
                    C.read(home/'transitions'/TRANSITION/'TRANSITION_PASS.json'),
                    C.sha(C.CODE/'LOCK.json'), C.read(C.CODE/'LOCK.json'), C.sha(C.ROOT/'SCOPE.json'))


def preserved_launch(case, count, home):
    label = 'r2s_' + case
    folder = C.ROOT/'submissions'/label
    ack_path = folder/'ACK.json'
    if not ack_path.exists() or C.read(ack_path)['spec']['source'] != PARENT_SOURCE:
        return False
    C.require(C.ROOT == home/'solve', 'Existing worker outside recovery solve scope')
    launch = C.read(C.ROOT/'cases'/case/'LAUNCHED.json')
    ack = C.read(ack_path)
    array = '0' if count == 1 else f'0-{count-1}'
    expected = S.spec(label, 'worker', [case], array=array, cpus=2, memory=18432, minutes=120)
    expected.update(source=PARENT_SOURCE, script=str(PARENT_CODE/'job.sh'))
    C.require(ack['job'] == launch['job'] and ack['spec'] == expected and launch['packets'] == count,
              'Existing numerical assignment changed')
    C.require(C.read(folder/'RELEASED.json') == dict(job=ack['job'], spec_sha256=C.digest(expected)),
              'Existing worker release uncertain')
    old_gate = C.read(C.ROOT/'submissions'/(label+'_gate')/'ACK.json')
    old_expected = S.spec(label+'_gate', 'recovery', ['gate', case], dependency=launch['job'], minutes=15)
    old_expected.update(source=PARENT_SOURCE, script=str(PARENT_CODE/'job.sh'))
    C.require(old_gate == dict(job=launch['gate'], spec=old_expected), 'Existing gate identity changed')
    mapping = C.read(home/ADOPTION)['gates'][case]
    C.require(mapping['worker'] == launch['job'] and mapping['old_gate'] == launch['gate'],
              'Replacement gate does not cover existing assignment')
    replacement = C.ROOT/'submissions'/(label+'_window_v4_gate')
    spec = S.spec(label+'_window_v4_gate', 'recovery', ['gate', case], dependency=launch['job'], minutes=15)
    C.require(C.read(replacement/'ACK.json') == dict(job=mapping['new_gate'], spec=spec),
              'Replacement gate ACK changed')
    C.require(C.read(replacement/'RELEASED.json') == dict(job=mapping['new_gate'], spec_sha256=C.digest(spec)),
              'Replacement handoff release uncertain')
    return True
