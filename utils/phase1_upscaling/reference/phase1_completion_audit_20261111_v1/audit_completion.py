"""Read-only, bounded final census of immutable case-completion receipts."""
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tomllib

BASE = Path('/orcd/data/juanes/001/shaowen')
P = BASE / 'phase1_pc40_adaptive10_campaign_20261030_v1'
C = BASE / 'phase1_pc40_adaptive10_runner_20261030_v5/configuration'
R = BASE / 'phase1_final_reference4_20261111_v2'
METHOD = 'ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
CONFIG = '7cbd42d913d516c765a6e3dff9f06994814e8b83ea67b2f96c98ae08289b1f36'
SOURCE = '8b61b7dabc42f57c858a2d137858fece866174ede388aa75fa6479137b2c0fbe'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def validate_cases(expected, scope_cases, markers, seals):
    require(len(expected) == len(set(expected)) == 2592, 'Frozen case count/uniqueness')
    require(len(scope_cases) == len(set(scope_cases)) == 2592, 'Production case uniqueness')
    require(set(expected) == set(scope_cases) == set(markers) == set(seals), 'Case coverage mismatch')
    for case in expected:
        m, s = markers[case], seals[case]
        require(m['case_id'] == case and m['positions'] == 522, 'Incomplete/wrong case')
        require(m['method'] == METHOD, 'Wrong scientific method')
        require(m['plan_sha256'] == s['plan_sha256'], 'Plan receipt mismatch')
    return len(expected) * 522


def main():
    lock_bytes = (C / 'LOCK.json').read_bytes()
    require(sha(lock_bytes) == CONFIG, 'Frozen configuration changed')
    catalog = (C / 'CASE_ASSIGNMENTS.csv').read_bytes()
    require(sha(catalog) == json.loads(lock_bytes)['files']['CASE_ASSIGNMENTS.csv'], 'Catalog changed')
    rows = list(csv.DictReader(io.StringIO(catalog.decode())))
    require(all(r['n_assignments'] == '522' and r['co2_brine_contact_angle_deg'] == '40'
                and r['clay_pc_sample_id'] in ('4', '6') for r in rows), 'Frozen design differs')
    scenarios = Counter(r['scenario_id'] for r in rows)
    require(len(scenarios) == 324 and set(scenarios.values()) == {8}, 'Scenario/realization count')
    expected = [r['case_id'] for r in rows]
    scope = read(P / 'SCOPE.json')
    require(scope['configuration_sha256'] == CONFIG, 'Production configuration differs')
    marker_files = list((P / 'complete_cases').glob('*.json'))
    marker_bytes = {f.stem: f.read_bytes() for f in marker_files}
    markers = {k: json.loads(v) for k, v in marker_bytes.items()}
    seals = {c: read(P / 'plans' / (c + '.seal.json')) for c in expected}
    positions = validate_cases(expected, scope['cases'], markers, seals)
    states = {}
    for rel, cursor in [('STATE.json', 2592), ('round2_20261109_v1/solve/STATE.json', 125),
                        ('round3_20261110_v1/solve/STATE.json', 55)]:
        states[rel] = read(P / rel)
        require(states[rel] == dict(active=[], cursor=cursor), 'Campaign controller active')
    require(sha((R / 'LOCK.json').read_bytes()) == SOURCE, 'Recovery source pin changed')
    batch = read(R / 'output/batches/0.json')
    require(batch['all_passed'] and batch['finished'] and not batch['held'], 'Recovery batch incomplete')
    keys = read(R / 'AUTHORIZATION.json')['keys']
    require(len(keys) == 4 and set(batch['published']) == set(keys), 'Final four coverage')
    require(batch['source'] == SOURCE and batch['job'] == '28695499', 'Publisher identity')
    final = []
    for key in keys:
        root = R / 'output' / key
        qpath = root / 'QUALIFIED_PASS.json'
        q, complete = read(qpath), read(root / 'COMPLETE.json')
        published = read(R / 'output/published' / key / 'PASS.json')
        case, slot = key.rsplit('_core', 1)
        require(complete['passed'] and complete['source'] == SOURCE
                and complete['qualified_pass_sha256'] == sha(qpath.read_bytes()), 'Recovery lifecycle')
        require(q['passed'] and q['key'] == key and q['source'] == SOURCE and q['method'] == METHOD, 'Recovery identity')
        require(q['completed_calls_repeated'] == q['saved_primary_solves_repeated'] == 0
                and q['new_reference_solves'] == 1, 'Saved work reuse')
        require(published['passed'] and published['source'] == SOURCE and published['new_solves'] == 0
                and published['new_complete_cases'] == [case], 'Publication incomplete')
        receipt = read(P / 'runs' / case / 'done' / ('core' + slot + '.json'))
        require(receipt == q['receipt'] == published['results'][0]['receipt'], 'Publication receipt mismatch')
        archive = P / 'results' / case / ('core' + slot + '.zip')
        require(archive.stat().st_size == receipt['bytes'], 'Archive size changed')
        d = tomllib.loads((root / 'work/recovery/precision/RESULT.toml').read_text())
        require(d['passed'] and d['new_primary_solves'] == 0 and d['new_reference_solves'] == 1
                and d['primary_target'] == 1e-9 and d['reference_target'] == 1e-11
                and d['agreement_target'] == 1e-5, 'Scientific reference check')
        final.append(dict(key=key, receipt=receipt, reference_seconds=d['attempts'][1]['seconds'],
                          relative_flow_difference=d['difference'], saved_flow_calls=q['saved_calls_reused'],
                          new_later_flow_calls=q['new_flow_calls'], saved_primary_solves_repeated=0,
                          completion_seconds=complete['seconds']))
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', '28695498,28695499',
        '--format=JobID%40,State,ExitCode,Elapsed'], text=True)
    records = [line.split('|') for line in accounting.splitlines() if line.strip()]
    require({r[0] for r in records} == {'28695498_' + str(i) for i in range(4)} | {'28695499'}, 'Missing final accounting')
    require(all(r[1] == 'COMPLETED' and r[2] == '0:0' for r in records), 'Non-successful terminal job')
    # This is receipt-based closure, not a fresh rehash of 1.35 million accepted archives.
    proof = {c: sha(marker_bytes[c]) for c in sorted(marker_bytes)}
    result = dict(passed=True, utc=datetime.now(timezone.utc).isoformat(), campaign=str(P),
        configuration_sha256=CONFIG, method=METHOD, cases_complete=2592, cases_expected=2592,
        positions_complete=positions, remaining_cases=[], remaining_positions=0,
        completion_receipts_sha256=sha(json.dumps(proof, sort_keys=True).encode()),
        basis='Exact frozen catalog coverage, canonical 522-position completion receipts and plan seals; final four full recovery/publication proofs.',
        fresh_campaign_archive_rehash=False, final_recovery_source=SOURCE, final_cores=final,
        terminal_accounting=accounting, controller_states=states)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
