import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (root/'LOCK.json').exists()
assert sha(root/'GUARD_PARENT_LOCK.json')=='e39642918493925c6c122f783be2cd35e153fc51db9838ab5169f8241e95d613'
assert sha(root/'QUALIFIED_GUARD_LOCK.json')=='7cb711ca387ca448767710b6af9a1eafc6f3285e212c4e9e13a1dc5265e4d768'
parent=json.loads((root/'GUARD_PARENT_LOCK.json').read_text());guard=json.loads((root/'QUALIFIED_GUARD_LOCK.json').read_text())
allowed={'worker.py','recovery.py','job.sh','test_recovery.py','qualify_timeout.py'}
for name,digest in parent['code'].items():
    if name not in allowed:assert sha(root/name)==digest,name
for name in ('worker.py','allocation_guard.py','test_allocation_guard.py'):assert sha(root/name)==guard['code'][name],name
assert sha(root/'science_bundle.zip')==parent['science_bundle_sha256']
extra={'GUARD_PARENT_LOCK.json','QUALIFIED_GUARD_LOCK.json','allocation_guard.py','test_allocation_guard.py','transition_contract.py','adopt_budget.py','test_adoption.py','build_adoption.py','seal_adoption.py'}
lock=dict(parent,code={n:sha(root/n) for n in sorted(set(parent['code'])|extra)},predecessor=sha(root/'GUARD_PARENT_LOCK.json'),
    guard_qualified_source=sha(root/'QUALIFIED_GUARD_LOCK.json'),future_recovery_adoption=True,production_adopted=False)
with (root/'LOCK.json').open('x') as f:json.dump(lock,f,indent=2,sort_keys=True)
print('SEALED_SUCCESSOR',sha(root/'LOCK.json'))
