"""Build a successor without editing either frozen predecessor."""
import hashlib
import json
from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
parent=root.parent/'phase1_round2_20261109_v2'
fix=root.parent/'phase1_round2_budget_fix_20261109_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'LOCK.json')=='e39642918493925c6c122f783be2cd35e153fc51db9838ab5169f8241e95d613'
assert sha(fix/'LOCK.json')=='7cb711ca387ca448767710b6af9a1eafc6f3285e212c4e9e13a1dc5265e4d768'
assert not (root/'LOCK.json').exists()
lock=json.loads((parent/'LOCK.json').read_text())
for name,digest in lock['code'].items():
    dest=root/name;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(parent/name,dest);assert sha(dest)==digest
for name in ['science_bundle.zip',*['plans/'+c+'.json' for c in lock['plans']]]:
    dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(parent/name,dest)
for name in ('allocation_guard.py','test_allocation_guard.py'):
    shutil.copyfile(fix/name,root/name)
shutil.copyfile(parent/'LOCK.json',root/'GUARD_PARENT_LOCK.json')
shutil.copyfile(fix/'LOCK.json',root/'QUALIFIED_GUARD_LOCK.json')
print('COPIED_UNCHANGED_PARENTS')
