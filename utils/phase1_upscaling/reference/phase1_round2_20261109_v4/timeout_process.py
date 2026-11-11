"""Bound stage execution by the allocation; never retry a mutating stage."""
import datetime
import os
from pathlib import Path
import signal
import subprocess
import time


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, end_epoch, reserve=600, exit_margin=30, clock=time.time):
        self.end = float(end_epoch)
        self.reserve = reserve
        self.exit_margin = exit_margin
        self.clock = clock

    @classmethod
    def allocation(cls, fields):
        # Slurm reports EndTime in the node's local timezone, like datetime.timestamp.
        return cls(datetime.datetime.fromisoformat(fields['EndTime']).timestamp())

    def limit(self, finalizing=False):
        remaining = self.end - self.clock() - self.exit_margin
        if not finalizing:
            remaining -= self.reserve
        if remaining <= 0:
            raise BudgetExhausted('No stage time remains within the allocation')
        return min(6900, remaining)


def live_group(pgid):
    found = subprocess.run(['pgrep', '-g', str(pgid)], capture_output=True, text=True, timeout=5)
    if found.returncode not in (0, 1):
        raise RuntimeError('Cannot establish child process-group terminality')
    live = []
    for pid in found.stdout.split():
        try:
            fields = Path('/proc', pid, 'stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            continue
        # Zombies cannot write outputs; do not confuse them with live descendants.
        if int(fields[2]) == pgid and fields[0] not in ('Z', 'X'):
            live.append(int(pid))
    return live


def terminate(p, grace=20):
    if os.name != 'posix':
        p.kill()
        p.wait(timeout=5)
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A parent may exit while descendants still run. Kill the whole owned group.
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait(timeout=5)
    until = time.monotonic() + 5
    while live_group(p.pid):
        if time.monotonic() >= until:
            raise RuntimeError('Child group is not terminal; finalization forbidden')
        time.sleep(.05)


def process(args, log, fd, env=None, cwd=None, timeout=6900):
    if timeout <= 0:
        raise BudgetExhausted('Stage not launched: non-positive time budget')
    start = time.monotonic()
    with log.open('x') as stream:
        p = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                             env=env, cwd=cwd,
                             pass_fds=(fd,) if fd is not None and os.name == 'posix' else (),
                             start_new_session=True)
        try:
            code = p.wait(timeout=timeout)
        except BaseException:
            try:
                terminate(p)
            except BaseException as error:
                # A wait/inspection timeout here is NOT a recoverable flow timeout.
                raise RuntimeError('Child not terminal; finalization forbidden') from error
            raise
    print('STAGE_EXIT', log.name, code, 'seconds', time.monotonic()-start, flush=True)
    return code
