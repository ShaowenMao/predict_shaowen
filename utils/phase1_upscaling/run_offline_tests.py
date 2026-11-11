"""Run selected local regressions in separate processes, never a campaign CLI."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from verify_package import verify

ROOT = Path(__file__).resolve().parent
SUITES = [
    ('.', ['test_package']),
    ('reference/phase1_pc40_adaptive10_runner_20261030_v5',
     ['test_runner', 'test_cache_integration', 'test_transition']),
    ('reference/phase1_round3_20261110_v1',
     ['test_allocation_guard', 'test_recovery', 'test_round3', 'test_timeout']),
    ('reference/phase1_final_reference4_20261111_v2', ['test_accounting', 'test_primary']),
    ('reference/phase1_completion_audit_20261111_v1', ['test_audit']),
]


def main():
    print(verify(ROOT), flush=True)
    # Original source names such as common/worker collide across versions.
    # Separate processes retain the original import contract without patching it.
    with tempfile.TemporaryDirectory(prefix='phase1_offline_') as tmp:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(('SLURM_', 'PHASE1_')) and k != 'PYTHONPATH'}
        env.update(PYTHONDONTWRITEBYTECODE='1', PHASE1_RUN_ROOT=tmp, USER='offline_test')
        for folder, modules in SUITES:
            print('\nSUITE ' + folder, flush=True)
            subprocess.run([sys.executable, '-B', '-m', 'unittest', *modules, '-v'],
                           cwd=ROOT / folder, env=env, check=True, timeout=180)
    print('\nAll selected offline suites passed.', flush=True)


if __name__ == '__main__':
    main()
