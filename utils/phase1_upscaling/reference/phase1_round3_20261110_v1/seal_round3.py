import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'LOCK.json').exists()
assert sha(root/'ROUND2_PARENT_LOCK.json') == '988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019'
parent = json.loads((root/'ROUND2_PARENT_LOCK.json').read_text())
allowed = {'recovery.py','cache_contract.py','timeout_process.py','test_timeout.py','job.sh','qualify_timeout.py','test_recovery.py','test_allocation_guard.py'}
for name, digest in parent['code'].items():
    if name not in allowed:
        assert sha(root/name) == digest, name
assert sha(root/'science_bundle.zip') == parent['science_bundle_sha256']
extra = {'ROUND2_PARENT_LOCK.json','round3_inventory.py','test_round3.py','build_round3.py','seal_round3.py'}
lock = dict(parent, code={n:sha(root/n) for n in sorted(set(parent['code'])|extra)},
    predecessor=sha(root/'ROUND2_PARENT_LOCK.json'), round_number=3, numerical_packet_size=1,
    numerical_minutes=240, stage_cap_seconds=13800, max_worker_reservations=128, production_adopted=False)
with (root/'LOCK.json').open('x') as f:
    json.dump(lock,f,indent=2,sort_keys=True)
print('SEALED_ROUND3',sha(root/'LOCK.json'))
