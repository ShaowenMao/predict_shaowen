import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
names=['exact_export.py','test_exact_export.py','export_saved.py','recover_saved.py','native_packet.jl','common.py','slurm.py','job.sh','seal.py','science_bundle.zip']
lock=dict(purpose='isolated_exact_spacing_export_qualification_only',production_adoption=False,code={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in names})
with (p/'LOCK.json').open('x') as f:json.dump(lock,f,sort_keys=True,indent=2)
print(hashlib.sha256((p/'LOCK.json').read_bytes()).hexdigest())
