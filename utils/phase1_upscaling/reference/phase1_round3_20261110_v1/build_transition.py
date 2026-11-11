"""Build a future-only routing successor; scientific worker body stays at v4."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'phase1_pc40_adaptive10_runner_20261030_v4'
BASE_PIN = '709f7267bd1233138a1d1e50e7999f536343f566ea01e51a352922398fdabbd8'
OLD_PIN = '4497359d39e01a47ce830ca4495a9682ba330b210eaa0856b785c45761d551c5'
REMOTE = '/orcd/data/juanes/001/shaowen/' + HERE.name

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def replace(s, before, after):
    assert s.count(before) == 1, before
    return s.replace(before, after)

def main():
    assert not (HERE / 'LOCK.json').exists()
    assert sha(BASE / 'LOCK.json') == BASE_PIN
    base = json.loads((BASE / 'LOCK.json').read_text())
    for n, h in base['code'].items():
        assert sha(BASE / n) == h, n
        p = HERE / n
        p.parent.mkdir(parents=True, exist_ok=True)
        assert not p.exists(), n
        shutil.copyfile(BASE / n, p)
    for name in ('plans', 'configuration'):
        shutil.copytree(BASE / name, HERE / name)
    shutil.copyfile(BASE / 'science_bundle.zip', HERE / 'science_bundle.zip')
    assert sha(HERE / 'science_bundle.zip') == base['science_bundle_sha256']
    p = HERE / 'worker.py'
    s = p.read_text()
    s = replace(s, "if __name__=='__main__':\n", "if __name__=='__main__':\n    from adoption import route_legacy_worker\n    route_legacy_worker('worker', sys.argv[1])\n")
    p.write_text(s, encoding='ascii', newline='\n')
    p = HERE / 'control.py'
    s = p.read_text()
    s = replace(s, "        C.require(a['approved_by_user'] is True and a['scope_sha256']==C.digest(scope) and a['source_sha256']==C.sha(C.CODE/'LOCK.json'),'Launch authorization identity')", "        from adoption import authorized_source\n        C.require(a['approved_by_user'] is True and a['scope_sha256']==C.digest(scope) and authorized_source(a,scope),'Launch authorization identity')")
    s = replace(s, "def launch(scope,case):\n", "def launch(scope,case):\n    from adoption import preserve_legacy_assignment\n    if preserve_legacy_assignment(case):return\n")
    s = replace(s, "def fixture(case):\n", "def fixture(case):\n    from adoption import route_legacy_worker\n    route_legacy_worker('fixture',case)\n")
    p.write_text(s, encoding='ascii', newline='\n')
    p = HERE / 'job.sh'
    s = p.read_text()
    start = s.index('CODE="$(cd ')
    end = s.index('mode="', start)
    s = s[:start] + '# New submissions pin their runner; shared RUN routing cannot change them.\nCODE="' + REMOTE + '"\n' + s[end:]
    s = replace(s, 'case "$mode" in', 'case "$mode" in\ntransition) exec /usr/bin/python3.12 -B "$CODE/transition_operator.py" "$@" ;;')
    p.write_text(s, encoding='ascii', newline='\n')
    # The predecessor test asserted all scheduling files were identical. Only the
    # two explicitly reviewed entrypoints may differ in this successor.
    p = HERE / 'test_cache_integration.py'
    s = p.read_text().replace("'planning.py','control.py','slurm.py'", "'planning.py','slurm.py'")
    p.write_text(s, encoding='ascii', newline='\n')
    code = dict(base['code'])
    for p in HERE.iterdir():
        if p.suffix in ('.py', '.jl', '.m', '.sh'):
            code[p.name] = sha(p)
    for n in code:
        code[n] = sha(HERE / n)
    lock = dict(base, code=code, predecessor=BASE_PIN, legacy_source=OLD_PIN,
                legacy_code='/orcd/data/juanes/001/shaowen/phase1_pc40_adaptive10_runner_20261030_v3',
                full_campaign_authorized=False)
    with (HERE / 'LOCK.json').open('x') as f:
        json.dump(lock, f, sort_keys=True, indent=2)
    print('SEALED', sha(HERE / 'LOCK.json'))

if __name__ == '__main__':
    main()
