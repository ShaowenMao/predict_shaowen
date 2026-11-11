"""Isolated Linux/Slurm qualification runner. No SSH, sbatch or publisher API.

Default CLI action is inspection only. Execution requires --execute-qualification
inside an already allocated advanced-tier compute job. Native imports are not
batched or assigned smaller memory by this module.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import tomllib

HERE = Path(__file__).resolve().parent
ACCOUNT = 'mit_amf_advanced_cpu'
ROOT_PREFIX = '/orcd/data/juanes/001/shaowen/predict_shaowen/'
FLAGS = ['--startup-file=no', '--math-mode=ieee', '--check-bounds=yes']


def need(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def digest_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def new_json(path, value):
    path = Path(path)
    need(not path.exists(), 'Refuse receipt overwrite')
    temp = path.with_name(path.name + '.partial')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    # Hard-link commit is no-clobber, including under concurrent collectors.
    os.link(temp, path)
    temp.unlink()


def toml_write(path, scalars, cases=(), guards=None):
    def line(k, v):
        need(re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', k), 'Invalid TOML key')
        need(type(v) in (str, bool, int, float, list), 'Unsupported TOML value')
        return k + ' = ' + json.dumps(v, allow_nan=False) + '\n'
    with Path(path).open('x') as f:
        for k, v in scalars.items():
            f.write(line(k, v))
        for case in cases:
            f.write('\n[[cases]]\n')
            for k, v in case.items():
                f.write(line(k, v))
        if guards is not None:
            f.write('\n[batch_guard]\n')
            for k, v in guards.items():
                f.write(line(k, v))


def pin_files(mapping):
    need(bool(mapping), 'Missing input/source pins')
    for p, h in mapping.items():
        need(digest_ok(h) and sha(p) == h, 'Pinned file changed: ' + p)


def validate_spec(s):
    need(s['qualification_only'] is True, 'Campaign execution not approved')
    need(s['mode'] in ('bootstrap', 'batch'), 'Unknown stage')
    need(s['julia_version'] == '1.12.6', 'Unqualified Julia version')
    need(type(s['cpus']) is int and s['cpus'] == 2, 'Only 2-CPU numerical/cache pilot supported')
    need(type(s['memory_gib']) is int and 2 <= s['memory_gib'] <= 1984, 'Invalid memory')
    need(positive(s['deadline_unix']) and positive(s['timeout_seconds']), 'Missing time bound')
    need(s['timeout_seconds'] <= 10800, 'Unbounded qualification time')
    for field in ('output', 'project', 'julia', 'study_root'):
        p = PurePosixPath(s[field])
        need(p.is_absolute() and '..' not in p.parts, 'Unsafe path: ' + field)
    root, out = PurePosixPath(s['study_root']), PurePosixPath(s['output'])
    need(str(root).startswith(ROOT_PREFIX) and 'warm_worker_qualification' in root.parts,
         'Use a dedicated durable warm_worker_qualification study directory')
    need(out.is_relative_to(root) and out != root, 'Output must be a new isolated child')
    need(PurePosixPath(s['project']).is_relative_to(root),
         'Use an isolated copy of the pinned environment, never the production project')
    need(s['storage_access_verified'] is True, 'Storage not verified')
    need(s['maintenance_clear'] is True, 'Maintenance guard closed')
    need(isinstance(s['cpu_target'], str) and re.fullmatch('[A-Za-z0-9_,;=+.-]+', s['cpu_target']),
         'Explicit CPU target required')
    need(bool(s['pins']), 'Missing source/input pins')
    need(all(digest_ok(h) for h in s['pins'].values()), 'Invalid file hash')
    for n in ('Project.toml', 'Manifest.toml'):
        need(str(PurePosixPath(s['project']) / n) in s['pins'], 'Missing pinned ' + n)
    if s['mode'] == 'batch':
        cases = s['cases']
        need(len(cases) in (1, 2, 4, 8), 'Use 1/2/4/8 qualification lengths')
        keys = [c['numerical_key'] for c in cases]
        if s.get('retention_probe', False):
            need(len(cases) in (4, 8) and keys[0] == keys[-1] and len(set(keys)) >= 2,
                 'Retention pilot must revisit a sentinel after distinct cores')
        else:
            need(len(set(keys)) == len(cases), 'Duplicate numerical owner')
        need(positive(s['retained_rss_limit_gib']) and
             s['retained_rss_limit_gib'] <= s['memory_gib'] - 1, 'Invalid retained-memory guard')
        need(positive(s['shutdown_margin_seconds']), 'Missing shutdown margin')
        for c in cases:
            need(type(c['slot']) is int and 1 <= c['slot'] <= 522, 'Invalid core slot')
            need(digest_ok(c['numerical_key']), 'Missing numerical identity')
            need(positive(c['seconds_upper']), 'Missing per-core upper time estimate')
            for p, h in ((c['raw'], c['raw_hash']), (c['geometry'], c['geometry_hash'])):
                need(s['pins'].get(p) == h and digest_ok(h), 'Input not pinned')
        need(sum(c['seconds_upper'] for c in cases) <= 3600, 'Batch exceeds one-hour compute bound')
        need(digest_ok(s['cache_receipt_sha256']), 'Cache receipt not pinned')
        need(digest_ok(s['pipeline_hash']), 'Missing pipeline identity')
        need(s['bridge'] in s['pins'], 'Bridge not pinned')
        need(s['claims_clear'] is True, 'Owner claim must be reconciled before testing')
        need(positive(s['claims_checked_unix']), 'Missing claim check timestamp')
    else:
        need(bool(s['source_depots']) and bool(s['load_only_files']), 'Missing bootstrap inputs')
        for p in s['load_only_files']:
            need(p in s['pins'], 'Unpinned bootstrap load file')
    return s


def validate_allocation(raw, s, job):
    fields = dict(t.split('=', 1) for t in raw.split() if '=' in t)
    expected = dict(JobId=job, Account=ACCOUNT, QOS=ACCOUNT, Partition='mit_normal',
                    NumCPUs=str(s['cpus']), NumNodes='1', JobState='RUNNING')
    for key, value in expected.items():
        need(fields.get(key) == value, 'Slurm allocation mismatch: ' + key)
    requested = fields.get('MinMemoryNode', '')
    need(requested in (str(s['memory_gib']) + 'G', str(s['memory_gib'] * 1024) + 'M'),
         'Slurm memory mismatch')
    return fields


def find_job_cgroup(job, proc_text=None):
    need(job.isdigit(), 'Invalid Slurm job ID')
    text = Path('/proc/self/cgroup').read_text() if proc_text is None else proc_text
    choices = []
    for line in text.splitlines():
        _, controllers, path = line.split(':', 2)
        if controllers and 'memory' not in controllers.split(','):
            continue
        p = PurePosixPath(path)
        need('..' not in p.parts, 'Invalid cgroup path')
        pieces = list(p.parts[1:])
        token = 'job_' + job
        if token in pieces:
            base = PurePosixPath('/sys/fs/cgroup/memory' if controllers else '/sys/fs/cgroup')
            choices.append(base.joinpath(*pieces[:pieces.index(token) + 1]))
    need(len(choices) == 1, 'Cannot identify the whole-job memory cgroup')
    return choices[0]


def memory(root):
    root = Path(root)
    if (root / 'memory.peak').exists():
        events = dict(line.split() for line in (root / 'memory.events').read_text().splitlines())
        result = dict(peak_bytes=int((root / 'memory.peak').read_text()),
                      limit_bytes=int((root / 'memory.max').read_text()),
                      events=sum(int(events.get(k, 0)) for k in ('max', 'oom', 'oom_kill')), version=2)
    else:
        result = dict(peak_bytes=int((root / 'memory.max_usage_in_bytes').read_text()),
                      limit_bytes=int((root / 'memory.limit_in_bytes').read_text()),
                      events=int((root / 'memory.failcnt').read_text()), version=1)
    return result


def memory_pass(m, gib):
    return (m['events'] == 0 and m['peak_bytes'] >= 0 and m['limit_bytes'] == gib * 1024**3
            and max(m['peak_bytes'] + 1024**3, 1.3 * m['peak_bytes']) <= m['limit_bytes'])


def julia_command(s, script, config, strict):
    args = [s['julia'], *FLAGS, '--threads=' + str(s['cpus']), '--project=' + s['project'],
            '--cpu-target=' + s['cpu_target']]
    if strict:
        args += ['--compiled-modules=strict', '--pkgimages=existing']
    return args + [str(script), str(config)]


def environment(s, depots):
    env = dict(os.environ)
    env.update(JULIA_DEPOT_PATH=':'.join(depots), JULIA_LOAD_PATH='@:@stdlib',
               JULIA_CPU_TARGET=s['cpu_target'], JULIA_PKG_PRECOMPILE_AUTO='0',
               JULIA_PKG_OFFLINE='true', JULIA_NUM_PRECOMPILE_TASKS='1',
               JULIA_NUM_THREADS=str(s['cpus']), OPENBLAS_NUM_THREADS=str(s['cpus']),
               OMP_NUM_THREADS=str(s['cpus']))
    return env


def supervise(args, env, out, name, cgroup, timeout):
    started = time.monotonic()
    with (out / (name + '.log')).open('x') as log, (out / (name + '_memory.jsonl')).open('x') as trace:
        process = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        try:
            while True:
                elapsed = time.monotonic() - started
                row = dict(seconds=elapsed, **memory(cgroup))
                trace.write(json.dumps(row) + '\n'); trace.flush()
                print(json.dumps(dict(stage=name, pid=process.pid, **row)), flush=True)
                if process.poll() is not None:
                    break
                if elapsed >= timeout:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(30, max(0.1, timeout-elapsed)))
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        result = dict(returncode=process.returncode, timed_out=timed_out,
                      seconds=time.monotonic()-started, memory=memory(cgroup),
                      command=args, production_modified=False)
        new_json(out / (name + '_execution.json'), result)
        return result


def cache_inventory(depot):
    # Once at bootstrap, not once per core. Bulk source/data trees are not copied.
    folder = Path(depot) / 'compiled'
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def stage_package_links(depot, sources):
    """Reuse installed data, but NEVER inherit another depot's compiled cache.

    Only new links in the isolated depot are created. Source depots are read-only
    inputs, and first-depot precedence matches Julia's usual package lookup.
    """
    depot = Path(depot)
    need(depot.is_dir() and not any(depot.iterdir()), 'Asset staging requires a new empty depot')
    linked = {}
    for source in sources:
        source = Path(source).resolve(strict=True)
        packages = source / 'packages'
        if packages.is_dir():
            for package in sorted(packages.iterdir()):
                if not package.is_dir(): continue
                for version in sorted(package.iterdir()):
                    if not version.is_dir(): continue
                    dest = depot / 'packages' / package.name / version.name
                    if dest.exists() or dest.is_symlink(): continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.symlink_to(version.resolve(), target_is_directory=True)
                    linked[str(dest.relative_to(depot))] = str(version.resolve())
        artifacts = source / 'artifacts'
        if artifacts.is_dir():
            for artifact in sorted(artifacts.iterdir()):
                if not artifact.is_dir() or not re.fullmatch('[0-9a-f]{40}', artifact.name): continue
                dest = depot / 'artifacts' / artifact.name
                if dest.exists() or dest.is_symlink(): continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(artifact.resolve(), target_is_directory=True)
                linked[str(dest.relative_to(depot))] = str(artifact.resolve())
    return linked


def cache_validate(receipt, s):
    need(receipt['status'] == 'PASS_STRICT_PACKAGE_LOAD_ONLY', 'Cache bootstrap not qualified')
    need(receipt['julia_version'] == s['julia_version'], 'Cache Julia mismatch')
    need(receipt['cpu_target'] == s['cpu_target'], 'Cache CPU target mismatch')
    need(receipt['project'] == s['project'], 'Cache project mismatch')
    need(receipt['release_id'] == s['release_id'], 'Cache execution source mismatch')
    for path, h in receipt['pins'].items():
        need(s['pins'].get(path) == h, 'Bootstrap pin not preserved: ' + path)
    need(receipt['compiler_flags'] == FLAGS and receipt['cpus'] == s['cpus'],
         'Compilation option/thread mismatch')
    need(receipt['memory_passed'] is True, 'Cache bootstrap memory not qualified')
    need(len(receipt['depots']) == 1, 'Shared fallback depots are prohibited')
    return receipt['depots']


def bootstrap_identity(s):
    body = {k:s[k] for k in ('release_id', 'julia_version', 'cpu_target', 'cpus', 'project', 'pins')}
    body['compiler_flags'] = FLAGS
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def claim_bootstrap(root, s, job, output):
    claim = Path(root) / 'cache_build_claims' / bootstrap_identity(s)
    claim.parent.mkdir(parents=True, exist_ok=True)
    # A persistent claim also survives failure: never steal a possibly live build.
    claim.mkdir(exist_ok=False)
    new_json(claim / 'intent.json', dict(job=job, output=str(output), status='BUILD_CLAIMED'))
    return claim


def collect_prefix(output, cases):
    """Read-only integrity collection; never mark outputs campaign-accepted."""
    output = Path(output)
    rows = []
    for index, case in enumerate(cases, 1):
        path = output / f'pass{index}_handoff.toml'
        if not path.exists():
            rows.append(dict(numerical_key=case['numerical_key'],
                             action='RECONCILE_PARTIAL_OUTPUT_AND_CLAIM_NO_AUTOMATIC_RERUN'))
            continue
        with path.open('rb') as f:
            r = tomllib.load(f)
        need(r['status'] == 'NUMERICAL_COMPLETE_REVIEW_REQUIRED', 'Wrong handoff status')
        need(r['numerical_key'] == case['numerical_key'], 'Wrong owner receipt')
        need(r['raw_hash'] == case['raw_hash'] and r['geometry_hash'] == case['geometry_hash'],
             'Wrong input identity')
        folder = output / f'pass{index:02}'
        for name, h in r['artifacts'].items():
            p = (folder / name).resolve()
            need(p.is_relative_to(folder.resolve()), 'Artifact escapes result folder')
            need(digest_ok(h) and sha(p) == h, 'Changed numerical output')
        need(sha(output / f'pass{index}_timing.toml') == r['timing_sha256'], 'Changed timing receipt')
        rows.append(dict(numerical_key=case['numerical_key'], resource_passed=r['resource_passed'],
                         action='PRESERVE_NUMERICS_RUN_UNCHANGED_REVIEW_EXPORT_IMPORT',
                         campaign_accepted=False))
    return rows


def execute(spec_file):
    driver_started = time.monotonic()
    s = validate_spec(read(spec_file))
    need(sys.platform.startswith('linux'), 'Execution must be on Engaging Linux')
    job = os.environ.get('SLURM_JOB_ID', '')
    need(job.isdigit(), 'No Slurm compute allocation; do not run on a login node')
    need(time.time() < s['deadline_unix'], 'Maintenance/wall-time guard closed')
    if s['mode'] == 'batch':
        need(0 <= time.time()-s['claims_checked_unix'] <= 300, 'Claim inventory is stale')
    raw = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    allocation = validate_allocation(raw, s, job)
    root, out = Path(s['study_root']).resolve(), Path(s['output']).resolve()
    need(str(root).startswith(ROOT_PREFIX) and out.is_relative_to(root) and out != root,
         'Resolved output outside isolated study')
    need(not out.exists(), 'Never overwrite or resume an old output directory')
    need(Path(s['project']).resolve().is_relative_to(root), 'Project resolves outside isolated study')
    pin_files(s['pins'])
    lock = read(HERE / 'source_lock.json')
    need(lock['identity'] == s['release_id'], 'Wrong execution release')
    pin_files({str(HERE / n): h for n, h in lock['files'].items()})
    cg = find_job_cgroup(job)
    need(memory_pass(memory(cg), s['memory_gib']), 'Allocation/cgroup headroom mismatch')
    out.mkdir(parents=True, exist_ok=False)
    new_json(out / 'allocation.json', dict(scontrol=allocation, cgroup=str(cg), spec_sha256=sha(spec_file)))
    timeout = min(s['timeout_seconds'], s['deadline_unix']-time.time()-60)
    need(timeout > 0, 'Insufficient time to start')
    if s['mode'] == 'bootstrap':
        claim = claim_bootstrap(root, s, job, out)
        depot = out / 'depot'; depot.mkdir()
        links = stage_package_links(depot, s['source_depots'])
        new_json(out / 'installed_asset_links.json', links)
        depots = [str(depot)]
        env = environment(s, depots)
        cfg = out / 'bootstrap.toml'
        toml_write(cfg, dict(job_id=job, project_file=str(Path(s['project']) / 'Project.toml'),
                   cpu_target=s['cpu_target'], receipt=str(out / 'bootstrap_result.toml'),
                   pinned_files=list(s['pins']), load_only_files=s['load_only_files']))
        run = supervise(julia_command(s, HERE / 'bootstrap.jl', cfg, False), env, out, 'bootstrap', cg, timeout)
        need(run['returncode'] == 0 and not run['timed_out'], 'Bootstrap incomplete; preserve diagnostics')
        before = cache_inventory(depot)
        # New process, same options, same modules, no package precompile allowed.
        load = out / 'strict_load.jl'
        with load.open('x') as f:
            f.write('using TOML\n@assert Sys.islinux() && string(VERSION)=="1.12.6"\n')
            for p in s['load_only_files']:
                f.write('include(' + json.dumps(p) + ')\n')
            f.write('println("STRICT_LOAD_PASSED")\n')
        remaining = min(s['timeout_seconds']-run['seconds'], s['deadline_unix']-time.time()-60)
        need(remaining > 0, 'No time for strict-load proof')
        probe = supervise(julia_command(s, load, cfg, True), env, out, 'strict_load', cg, remaining)
        need(probe['returncode'] == 0 and not probe['timed_out'], 'Strict load failed; do not fall back')
        need(before == cache_inventory(depot), 'Strict load unexpectedly changed package cache')
        pin_files(s['pins'])
        with (out / 'bootstrap_result.toml').open('rb') as f:
            details = tomllib.load(f)
        result = dict(status='PASS_STRICT_PACKAGE_LOAD_ONLY', julia_version=s['julia_version'],
                      cpu_target=s['cpu_target'], project=s['project'], depots=depots,
                      pins=s['pins'], packages=details['packages'], cache_files=before,
                      installed_asset_links=links,
                      release_id=s['release_id'], compiler_flags=FLAGS, cpus=s['cpus'],
                      memory_passed=memory_pass(memory(cg), s['memory_gib']),
                      numerical_equivalence_qualified=False, native_small_memory_approved=False,
                      campaign_approved=False)
        new_json(out / 'CACHE_RECEIPT.json', result)
        new_json(claim / 'receipt.json', dict(path=str(out / 'CACHE_RECEIPT.json'),
                                             sha256=sha(out / 'CACHE_RECEIPT.json')))
    else:
        need(sha(s['cache_receipt']) == s['cache_receipt_sha256'], 'Cache receipt changed')
        cache = read(s['cache_receipt'])
        depots = cache_validate(cache, s)
        # Shared cache integrity is rechecked once per qualification job, not per core.
        need(cache_inventory(depots[0]) == cache['cache_files'], 'Compiled cache drift')
        cfg = out / 'managed.toml'
        guards = dict(qualification_only=True, job_id=job, cgroup=str(cg),
                      retention_probe=s.get('retention_probe', False),
                      deadline_unix=min(s['deadline_unix'], time.time()+timeout),
                      memory_gib=s['memory_gib'], shutdown_margin_seconds=s['shutdown_margin_seconds'],
                      retained_rss_limit_gib=s['retained_rss_limit_gib'],
                      release_id=s['release_id'], config_sha256=sha(spec_file))
        toml_write(cfg, dict(pipeline=s['pipeline'], pipeline_hash=s['pipeline_hash'], bridge=s['bridge'],
                            threads=s['cpus'], passes=len(s['cases']), output=str(out / 'managed')),
                   cases=s['cases'], guards=guards)
        run = supervise(julia_command(s, HERE / 'worker.jl', cfg, True), environment(s, depots),
                        out, 'batch', cg, timeout)
        prefix = collect_prefix(out / 'managed', s['cases'])
        new_json(out / 'PRESERVED_CORES.json', dict(cores=prefix, execution=run,
                 campaign_approved=False, completed_work_recomputed=False))
        pin_files(s['pins'])
        need(cache_inventory(depots[0]) == cache['cache_files'], 'Compiled cache drift during batch')
        need(run['returncode'] == 0 and not run['timed_out'], 'Incomplete pilot; preserve prefix')
    new_json(out / 'QUALIFICATION_RUNNER_COMPLETE.json',
             dict(total_driver_seconds=time.monotonic()-driver_started,
                  whole_job_memory=memory(cg), campaign_approved=False,
                  scientific_comparison_pending=True, export_import_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('spec')
    parser.add_argument('--execute-qualification', action='store_true')
    args = parser.parse_args()
    if args.execute_qualification:
        execute(args.spec)
    else:
        s = validate_spec(read(args.spec))
        print(json.dumps(dict(valid_spec=True, mode=s['mode'], no_execution=True,
                              campaign_approved=False)))
