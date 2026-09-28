"""Run the preserved tiny fixtures, never campaign inputs or accepted science."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow as W


def main():
    p = argparse.ArgumentParser()
    p.add_argument('out', type=Path)
    p.add_argument('--julia', default='julia')
    a = p.parse_args()
    W.verify()
    version = subprocess.check_output([a.julia, '--version'], text=True).strip()
    if version != 'julia version 1.12.6':
        raise ValueError('Tests require the qualified Julia 1.12.6, found ' + version)
    out = W.fresh(a.out)
    snap = W.SNAP
    env = dict(os.environ, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', JULIA_NUM_THREADS='2')
    base = [a.julia, '--startup-file=no', '--math-mode=ieee', '--check-bounds=yes', '--threads=2']
    bridge = out / 'bridge' / ('phase_bridge.dll' if os.name == 'nt' else 'libphase_bridge.so')
    raw, geometry = [snap / 'frozen_preparation/tests' / n for n in ('raw.bin', 'geometry.bin')]
    commands = [
        ('pc', base + [str(snap / 'pipeline/vendor/pc/test/runtests.jl')]),
        ('bridge', [sys.executable, str(snap / 'pipeline/vendor/kr/build_bridge.py'),
                    str(snap / 'pipeline/vendor/numeric'), str(out / 'bridge')]),
        ('plan', base + [str(snap / 'frozen_preparation/test_plan.jl'), str(raw), str(geometry), str(out / 'planned')]),
        ('coupled', base + [str(snap / 'pipeline/test_pipeline.jl'), str(out / 'planned/raw.bin'),
                            str(bridge), str(out / 'flow')]),
        ('backend', base + [str(W.ROOT / 'tests/test_backend.jl'), str(out / 'planned/raw.bin'),
                            str(out / 'planned/geometry.bin'), str(bridge)]),
    ]
    for name, command in commands:
        print('Testing ' + name, flush=True)
        W.run(command, out / (name + '.log'), env)
    W.verify()
    W.write(out / 'PASS.json', dict(passed=True, platform=platform.platform(), julia=version,
            source_manifest_sha256=W.sha(W.ROOT / 'SOURCE_MANIFEST.json'), commands=commands,
            logs={name: W.sha(out / (name + '.log')) for name, _ in commands},
            scope='Small synthetic fixtures only; not full-case Linux deployment or reservoir acceptance',
            accepted_science_repeated=False))
    print('All preserved numerical fixture tests passed')


if __name__ == '__main__':
    main()
