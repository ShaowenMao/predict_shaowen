"""Copy the sealed predecessor; subsequent edits are isolated from production."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root.with_name('phase1_round2_20261109_v3')
pin = 'facf4d6fb3f93f144c19dfd6862e7cf45c24803b6333b4d69d6031304adc2c6a'
assert hashlib.sha256((parent/'LOCK.json').read_bytes()).hexdigest() == pin
lock = json.loads((parent/'LOCK.json').read_text())
for name, digest in lock['code'].items():
    source = parent/name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
    target = root/name
    assert not target.exists(), name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
for name in ['science_bundle.zip']:
    assert not (root/name).exists()
    shutil.copyfile(parent/name, root/name)
shutil.copyfile(parent/'LOCK.json', root/'WINDOW_PARENT_LOCK.json')
print('COPIED_SEALED_PARENT', pin)
