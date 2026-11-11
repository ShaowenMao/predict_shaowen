"""Select one exact running task; never merge records returned for an array."""
import os
import re
import subprocess


def identity(env):
    job = env.get('SLURM_JOB_ID', '')
    array = env.get('SLURM_ARRAY_JOB_ID')
    task = env.get('SLURM_ARRAY_TASK_ID')
    if not re.fullmatch(r'[0-9]+', job):
        raise RuntimeError('Missing or invalid Slurm job identity')
    if (array is None) != (task is None):
        raise RuntimeError('Incomplete Slurm array identity')
    if array is not None and not all(re.fullmatch(r'[0-9]+', x) for x in (array, task)):
        raise RuntimeError('Invalid Slurm array identity')
    return job, array, task


def select_record(raw, env):
    job, array, task = identity(env)
    records = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if not line.startswith('JobId=') or sum(x.startswith('JobId=') for x in line.split()) != 1:
            raise RuntimeError('Malformed allocation record')
        fields = dict(x.split('=', 1) for x in line.split() if '=' in x)
        if fields.get('JobId') == job:
            records.append(fields)
    if len(records) != 1:
        raise RuntimeError('Missing or ambiguous exact-task allocation')
    fields = records[0]
    if array is None:
        if 'ArrayJobId' in fields or 'ArrayTaskId' in fields:
            raise RuntimeError('Unexpected array allocation')
    elif fields.get('ArrayJobId') != array or fields.get('ArrayTaskId') != task:
        raise RuntimeError('Array allocation identity mismatch')
    if fields.get('JobState') != 'RUNNING':
        raise RuntimeError('Exact task is not running')
    return fields


def read_allocation(env=None, query=None):
    env = os.environ if env is None else env
    job, array, task = identity(env)
    target = job if array is None else array + '_' + task
    if query is None:
        query = lambda args: subprocess.check_output(args, text=True, timeout=90)
    return select_record(query(['scontrol', 'show', 'job', '-o', target]), env)
