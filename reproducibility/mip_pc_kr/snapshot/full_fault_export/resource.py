"""Small whole-job memory/allocation reader, including the launcher process."""
from pathlib import Path
from decimal import Decimal
import re


def parse_allocation(text, cpus, mib):
    f = dict(v.split('=', 1) for v in text.split() if '=' in v)
    for k, value in dict(Account='mit_amf_advanced_cpu', QOS='mit_amf_advanced_cpu',
                         Partition='mit_normal', NumCPUs=str(cpus)).items():
        assert f.get(k) == value, (k, f.get(k), value)
    assert f.get('NumNodes') in ('1', '1-1')
    memory = f['MinMemoryNode']
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([MG]?)', memory)
    assert match, memory
    actual = Decimal(match[1]) * (1024 if match[2] == 'G' else 1)
    assert actual == mib, (actual, mib)
    return f


def cgroup(job):
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, p = line.split(':', 2)
        parts = Path(p).parts
        name = 'job_' + str(job)
        if name in parts and (not controllers or 'memory' in controllers.split(',')):
            base = Path('/sys/fs/cgroup/memory' if controllers else '/sys/fs/cgroup')
            return base.joinpath(*parts[1:parts.index(name)+1])
    raise RuntimeError('Cannot locate whole-job memory cgroup')


def snapshot(root):
    root = Path(root)
    if (root/'memory.peak').exists():
        ev = dict(x.split() for x in (root/'memory.events').read_text().splitlines())
        return dict(peak_bytes=int((root/'memory.peak').read_text()),
                    limit_bytes=int((root/'memory.max').read_text()),
                    events=sum(int(ev.get(k, 0)) for k in ('max', 'oom', 'oom_kill')))
    return dict(peak_bytes=int((root/'memory.max_usage_in_bytes').read_text()),
                limit_bytes=int((root/'memory.limit_in_bytes').read_text()),
                events=int((root/'memory.failcnt').read_text()))


def passed(d, mib):
    return (d['events'] == 0 and d['limit_bytes'] == mib*1024**2
            and 0 < 21*d['peak_bytes'] <= 20*d['limit_bytes'])
