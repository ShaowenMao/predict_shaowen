"""Stage an unsealed derivative of the qualified timeout runner."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root.parent / 'phase1_timeout_candidate_20261031_v2'
assert hashlib.sha256((parent / 'LOCK.json').read_bytes()).hexdigest() == '86bfdbe7a6ccf43caec864a400b6b220f42d1fa089609fb97aeed47f71549336'
lock = json.loads((parent / 'LOCK.json').read_text())
for name in list(lock['code']) + ['science_bundle.zip', 'RUNTIME_BUNDLE.json', 'runtime_inventory.json', 'DATA_PINS.json', 'EXECUTABLE_PINS.json']:
    src = parent / name
    dst = root / name
    if dst.exists():
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
shutil.copyfile(parent / 'LOCK.json', root / 'TIMEOUT_PARENT_LOCK.json')
print('Staged unsealed derivative; source edits and tests required before sealing.')
