"""Compute-node verification of the legacy runtimes used read-only by this pilot."""
import hashlib,json,tomllib
from pathlib import Path
from pin_preflight import check
ROOT=Path(__file__).resolve().parent
check(tomllib.loads((ROOT/'runtime.toml').read_text()),remote=True)
lock=json.loads((ROOT/'data/old_launch_lock.json').read_text())
selected={}
for field in ('worker_pins','external'):
    for p,h in lock[field].items():
        if any(s in p for s in ('/environment/','corrected_hysteresis_15case_20260911_v1/runtime/', '/interpolation.jl','libphase_bridge.so')):
            selected[p]=h
assert len(selected)>5
for p,h in selected.items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, p
with (ROOT/'RUNTIME_PASS.json').open('x') as f:
    json.dump(dict(passed=True,files=selected),f,indent=2)
print('RUNTIME_PASS',len(selected))
