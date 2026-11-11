"""Seal the private test candidate; does not publish an adoption or launch jobs."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
parent=json.loads((root/'PARENT_LOCK.json').read_text())
assert sha(root/'PARENT_LOCK.json')=='c5b9ae6c9e49f6ff8a21f66aaba7724235a761de63e03bd04c1a96e93dc1bccf'
lock=dict(parent)
allowed={'worker.py','cache_contract.py','job.sh'}
for n,h in parent['code'].items():
    if n not in allowed:assert sha(root/n)==h,n
assert sha(root/'science_bundle.zip')==parent['science_bundle_sha256']
extra=['PARENT_LOCK.json','timeout_process.py','test_timeout.py','qualify_timeout.py','seal_candidate.py']
lock.update(code={n:sha(root/n) for n in sorted(set(parent['code'])|set(extra))},
            predecessor=sha(root/'PARENT_LOCK.json'),qualification_only=True,
            full_campaign_authorized=False,production_adopted=False)
out=root/'LOCK.json'
old=json.loads(out.read_text())
if old.get('qualification_only'):
    assert old==lock,'Already sealed; create a new version instead'
else:
    out.write_text(json.dumps(lock,sort_keys=True,indent=2)+'\n',encoding='ascii')
print('SEALED',sha(out))
