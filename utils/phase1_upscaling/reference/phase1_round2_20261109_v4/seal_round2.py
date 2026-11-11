"""Seal after local tests; never edit a sealed candidate."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parent_dir = root.parent/'phase1_timeout_candidate_20261031_v2'
parent = json.loads((root/'TIMEOUT_PARENT_LOCK.json').read_text())
assert sha(root/'TIMEOUT_PARENT_LOCK.json') == '86bfdbe7a6ccf43caec864a400b6b220f42d1fa089609fb97aeed47f71549336'
assert not (root/'LOCK.json').exists(), 'Never reseal; create a new version'
for case, digest in parent['plans'].items():
    dest = root/'plans'/f'{case}.json'; dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(parent_dir/'plans'/dest.name, dest)
    assert sha(dest) == digest
allowed = {'worker.py', 'flow_packet.jl', 'job.sh', 'qualify_timeout.py', 'test_timeout.py'}
for name, digest in parent['code'].items():
    if name not in allowed:
        assert sha(root/name) == digest, name
old_flow = (parent_dir/'flow_packet.jl').read_text()
new_flow = (root/'flow_packet.jl').read_text()
guard = '    prior=joinpath(work,"PRIOR_INPUT_PASS.toml")\n    if isfile(prior)\n        @assert TOML.parsefile(prior)==proof "Round-two replay must match the exact original input"\n    end\n'
assert new_flow.replace(guard, '') == old_flow, 'Numerical body changed'
assert sha(root/'science_bundle.zip') == parent['science_bundle_sha256']
extra = {'build_base.py', 'recovery.py', 'test_recovery.py', 'seal_round2.py', 'TIMEOUT_PARENT_LOCK.json'}
lock = dict(parent)
lock.update(code={n:sha(root/n) for n in sorted(set(parent['code']) | extra)},
    round_two_only=True, qualification_only=False, full_campaign_authorized=False,
    production_adopted=False, predecessor=sha(root/'TIMEOUT_PARENT_LOCK.json'),
    recovery={'saved_first':True, 'interrupted_packet_size':1, 'unstarted_packet_size':8,
              'max_cases':8, 'max_worker_reservations':128, 'preserve_original_work':True})
with (root/'LOCK.json').open('x') as f:
    json.dump(lock, f, sort_keys=True, indent=2)
print('SEALED', sha(root/'LOCK.json'))
