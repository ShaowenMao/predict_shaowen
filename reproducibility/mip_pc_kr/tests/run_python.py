"""Run publication and preserved table tests with durable command/log evidence."""
import argparse
from pathlib import Path
import platform
import sys

import h5py
import numpy
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow as W


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    W.verify()
    out = W.fresh(args.out)
    vendor = W.SNAP / 'historical_releases/mip_matrix_launch_20260911_v1/vendor'
    suites = [('package', W.ROOT / 'tests', 'test_workflow.py'),
              ('physical', vendor, 'test_tables.py'),
              ('pc_onset', vendor, 'test_pc_only.py')]
    commands = []
    for name, source, pattern in suites:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(source), '-p', pattern, '-v']
        W.run(command, out / (name + '.log'))
        commands.append(command)
    W.verify()
    W.write(out / 'PASS.json', dict(passed=True, python=sys.version,
            platform=platform.platform(), libraries=dict(numpy=numpy.__version__,
            scipy=scipy.__version__, h5py=h5py.__version__), commands=commands,
            source_manifest_sha256=W.sha(W.ROOT / 'SOURCE_MANIFEST.json'),
            logs={name: W.sha(out / (name + '.log')) for name, _, _ in suites},
            scope='Synthetic fixtures and preservation guards; not live campaign acceptance',
            accepted_science_repeated=False))
    print('All Python publication tests passed')


if __name__ == '__main__':
    main()
