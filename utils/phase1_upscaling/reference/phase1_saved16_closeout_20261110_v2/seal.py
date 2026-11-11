import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
names=['closeout.py','test_closeout.py','common.py','slurm.py','job.sh','seal.py']
lock=dict(purpose='portable_cross_node_archive_preservation_guard_only',parent_source='5bcedae1b145ac77f0fb19975acc8ed7addcae4538c3d5ba546caea3d255bbd6',code={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names})
with (p/'LOCK.json').open('x') as f:json.dump(lock,f,sort_keys=True,indent=2)
print(hashlib.sha256((p/'LOCK.json').read_bytes()).hexdigest())
