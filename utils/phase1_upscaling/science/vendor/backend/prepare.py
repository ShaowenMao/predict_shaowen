"""Build an isolated execution adapter, never edit or submit the campaign."""
from pathlib import Path
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PARENT = STUDY / 'campaign_numerical_v1/releases/6c240610c9f90e81'
CONTRACT = STUDY / 'memory_batch_preparation_20260915_v1/results_v2/worker_contract.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def adapt(source):
    start = 'for pass in 1:config["passes"]\n'
    end = '\nend\nU.Diagnostics.verify_receipt!'
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError('Pinned worker structure changed; manual review required')
    prefix, rest = source.split(start)
    body, footer = rest.split(end)
    # The calculation body is copied unchanged; the function boundary releases
    # its per-core local references before the post-core GC/memory measurement.
    prefix = prefix.replace('config = TOML.parsefile(ARGS[1])\n',
        'config = TOML.parsefile(ARGS[1])\ninclude("BatchGuard.jl")\n'
        'const session = BatchGuard.Session(config)\n', 1)
    return (prefix + '\nfunction run_one_core(pass, config)\n'
            + '# BEGIN PINNED SCIENTIFIC BODY\n' + body
            + '\n# END PINNED SCIENTIFIC BODY\n    nothing\nend\n'
            + 'for pass in 1:config["passes"]\n'
            + '    BatchGuard.before!(session, pass)\n'
            + '    run_one_core(pass, config)\n'
            + '    BatchGuard.record!(session, pass, MemoryProbe.after_gc())\nend\n'
            + 'U.Diagnostics.verify_receipt!' + footer
            + '\nBatchGuard.finish!(session)\n')


def build(output=None):
    contract = json.loads(CONTRACT.read_text())
    sources = {k: STUDY / v['path'] for k, v in contract['source_bindings'].items()}
    for k, p in sources.items():
        if sha(p) != contract['source_bindings'][k]['sha256']:
            raise ValueError('Original source drift: ' + k)
    original = (PARENT / 'worker.jl').read_text()
    adapter = adapt(original)
    local = ['BatchGuard.jl', 'bootstrap.jl', 'runner.py', 'prepare.py']
    fingerprint = {
        'original_sources': contract['source_bindings'],
        'memory_probe': sha(PARENT / 'MemoryProbe.jl'),
        'local_sources': {n: sha(HERE / n) for n in local},
        'adapter_sha256': hashlib.sha256(adapter.encode()).hexdigest(),
    }
    identity = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    output = Path(output) if output else HERE / 'releases' / identity[:16]
    if output.exists():
        lock = json.loads((output / 'source_lock.json').read_text())
        if lock['identity'] != identity:
            raise ValueError('Refuse conflicting release overwrite')
        for n, h in lock['files'].items():
            if sha(output / n) != h:
                raise ValueError('Existing release modified: ' + n)
        return output
    output.mkdir(parents=True)
    (output / 'worker.jl').write_text(adapter, newline='\n')
    for n in ['Backend.jl', 'ManagedLU.jl', 'MemoryProbe.jl']:
        shutil.copyfile(PARENT / n, output / n)
    for n in local:
        shutil.copyfile(HERE / n, output / n)
    lock = dict(identity=identity, fingerprint=fingerprint,
                files={p.name: sha(p) for p in sorted(output.iterdir())},
                production_modified=False, engaging_qualified=False,
                campaign_launch_approved=False)
    (output / 'source_lock.json').write_text(json.dumps(lock, indent=2) + '\n', newline='\n')
    for k, p in sources.items():
        if sha(p) != contract['source_bindings'][k]['sha256']:
            raise ValueError('Original source changed during build: ' + k)
    return output


if __name__ == '__main__':
    print(build())
