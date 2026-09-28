"""Pure resource and reconciliation checks shared by the real dispatcher."""
import math


def memory_gib(text, cpus):
    text=text.strip();per_cpu=text.endswith('c')
    if text[-1:] in ('c','n'):text=text[:-1]
    factor={'K':1/1024**2,'M':1/1024,'G':1,'T':1024}[text[-1]]
    value=float(text[:-1])*factor*(cpus if per_cpu else 1)
    if not math.isfinite(value) or value<=0 or cpus<1:raise ValueError('Invalid allocation')
    return value


def unique_job(name, texts):
    found=set()
    for text in texts:
        for line in text.splitlines():
            p=line.split('|')
            if len(p)>1 and p[1]==name and p[0].isdigit():found.add(p[0])
    if len(found)>1:raise ValueError('Duplicate real submissions: '+name)
    return next(iter(found)) if found else None


def can_admit(cpu,mem):
    if not math.isfinite(cpu) or not math.isfinite(mem) or cpu<0 or mem<0:
        raise ValueError('Invalid reserved resource totals')
    return cpu+1<=500 and mem+2<=1984
