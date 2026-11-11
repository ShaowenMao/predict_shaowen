import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'LOCK.json').exists()
assert sha(root/'WINDOW_PARENT_LOCK.json') == 'facf4d6fb3f93f144c19dfd6862e7cf45c24803b6333b4d69d6031304adc2c6a'
parent = json.loads((root/'WINDOW_PARENT_LOCK.json').read_text())
allowed = {'recovery.py', 'job.sh', 'test_recovery.py', 'qualify_timeout.py'}
for name, digest in parent['code'].items():
    if name not in allowed:
        assert sha(root/name) == digest, name
assert sha(root/'science_bundle.zip') == parent['science_bundle_sha256']
extra = {'WINDOW_PARENT_LOCK.json','window_contract.py','adopt_window.py','test_window.py','build_window.py','seal_window.py'}
lock = dict(parent, code={n:sha(root/n) for n in sorted(set(parent['code'])|extra)},
            predecessor=sha(root/'WINDOW_PARENT_LOCK.json'), recovery_window_only=True,
            effective_solve_max_cases=128, max_worker_reservations=128, production_adopted=False)
with (root/'LOCK.json').open('x') as f:
    json.dump(lock, f, indent=2, sort_keys=True)
print('SEALED_WINDOW_SUCCESSOR', sha(root/'LOCK.json'))
