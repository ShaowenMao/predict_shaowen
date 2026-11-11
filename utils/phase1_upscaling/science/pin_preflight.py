"""Check configured runtime pins against the frozen lock, before any solve."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check(config, remote=False):
    lock = json.loads((ROOT / 'data/old_launch_lock.json').read_text())
    contract = lock['contract']
    names = {'bridge_sha256': 'bridge_sha256', 'pipeline_hash': 'pipeline_sha256',
             'kernel_sha256': 'kernel_sha256', 'selector_sha256': 'selector_sha256'}
    for key, source in names.items():
        assert re.fullmatch('[0-9a-f]{64}', config[key]), key
        assert config[key] == contract[source], key
    assert config['pilot_only'] and not config['full_campaign_authorized']
    assert lock['external'][config['bridge']] == config['bridge_sha256']
    assert sha(ROOT / 'vendor/kernel/Kernel.jl') == config['kernel_sha256']
    assert sha(ROOT / 'vendor/kernel/SelectStates.jl') == config['selector_sha256']
    pipeline = ROOT / 'vendor/pipeline'
    paths = sorted((p for p in pipeline.rglob('*') if p.is_file()
                    and '__pycache__' not in p.parts and p != pipeline / 'source_lock.json'),
                   key=lambda p: p.relative_to(pipeline).as_posix())
    lines = [p.relative_to(pipeline).as_posix() + ':' + sha(p) for p in paths]
    tree = hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    assert tree == config['pipeline_hash'], ('pipeline_tree', tree)
    if remote:
        assert sha(Path(config['bridge'])) == config['bridge_sha256']
    return {'passed': True, 'pins_checked': names, 'pipeline_files': len(paths),
            'external_bridge_verified': remote, 'full_campaign_authorized': False}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--report')
    args = parser.parse_args()
    config = tomllib.loads((ROOT / 'runtime.toml').read_text())
    report = check(config, args.remote)
    rejected = 0
    for key in ('bridge_sha256', 'pipeline_hash', 'kernel_sha256', 'selector_sha256'):
        for bad in ('0' * 64, config[key] + '0'):
            altered = dict(config, **{key: bad})
            try:
                check(altered)
            except AssertionError:
                rejected += 1
            else:
                raise AssertionError(('bad_pin_accepted', key))
    assert rejected == 8
    report['bad_pin_regression_tests_passed'] = rejected
    if args.report:
        with (ROOT / args.report).open('x') as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))

if __name__ == '__main__':
    main()
