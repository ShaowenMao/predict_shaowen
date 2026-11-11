"""Copy the sealed recovery predecessor into a separate round-three candidate."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root.with_name('phase1_round2_20261109_v4')
pin = '988abcc32f15861aa95e6662c932e28b897c4bd616f42893897d944cbaaff019'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(parent/'LOCK.json') == pin
lock = json.loads((parent/'LOCK.json').read_text())
files = dict(lock['code'])
files.update({f'plans/{case}.json': digest for case, digest in lock['plans'].items()})
for name, digest in files.items():
    source = parent/name
    assert sha(source) == digest, name
    target = root/name
    if target.exists():
        assert sha(target) == digest, name
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
shutil.copyfile(parent/'science_bundle.zip', root/'science_bundle.zip')
shutil.copyfile(parent/'LOCK.json', root/'ROUND2_PARENT_LOCK.json')
print('COPIED_ROUND2', pin)
