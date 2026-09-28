"""Create a fresh, hash-locked reservoir launcher without changing old artifacts."""
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import tomllib

SOURCE = Path(__file__).resolve().parent
BASE = Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
HYBRID = BASE / 'hybrid_pc60_production_kr_s03_c022_case12_20260911_v1'
OLD = HYBRID / 'reservoir'
NEW = HYBRID / 'reservoir_preflightfix1'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def checked_replace(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def main():
    subprocess.run(['sha256sum', '--check', '--quiet', 'STUDY_SHA256.txt'], cwd=OLD, check=True)
    subprocess.run(['sha256sum', '--check', '--quiet', 'INPUTS_SHA256.txt'], cwd=OLD, check=True)
    q = json.loads((HYBRID / 'qualified.json').read_text())
    assert q['passed']
    spec = HYBRID / 'export/candidate_specific_s03_c022_case12.mat'
    tables = HYBRID / 'export/candidate_tables_s03_c022_case12.mat'
    assert sha(spec) == q['specific_sha256'] and sha(tables) == q['table_sha256']
    assert not NEW.exists(), 'Recovery root exists; refusing to overwrite'
    assert shutil.disk_usage(HYBRID).free > 200 * 1024**3
    m = tomllib.loads((OLD / 'campaign.toml').read_text())
    assert m['cases'][0]['task'] == 1137 and len(m['cases']) == 1
    assert m['restart_reuse'] is False and m['production_promotion'] is False
    assert len(m['workflow']['retain_years']) == 210
    assert m['workflow']['qoi_schema_version'] == 4
    NEW.mkdir()
    for name in ('logs', 'tmp'):
        (NEW / name).mkdir()
    shutil.copytree(OLD / 'source', NEW / 'source')
    for name in ('pair_input_audit.jl', 'test_pair_input_audit.jl', 'stage_recovery.py', 'README.md'):
        shutil.copy2(SOURCE / name, NEW / 'source' / name)
    for name in ('INPUTS_SHA256.txt', 'comparison_contract.json'):
        shutil.copy2(OLD / name, NEW / name)
    guard = NEW / 'source/candidate_guard.sh'
    guard.write_text(checked_replace(guard.read_text(), str(OLD), str(NEW)))
    for suffix, kind in (('preflight', 'pf'), ('full', 'full')):
        script = NEW / f'source/gom_step62_effective_pc_global_plateau_{suffix}.sbatch'
        original = script.read_text()
        assert '#SBATCH --cpus-per-task=8' in original
        assert f'#SBATCH --mem={24 if suffix == "preflight" else 18}G' in original
        script.write_text(checked_replace(original, f'--job-name=cand_s03c022_{kind}',
                                         f'--job-name=hyb60_s03c022_{kind}_fix1'))
    new_id = m['campaign_id'] + '_preflightfix1'
    manifest = NEW / 'campaign.toml'
    text = checked_replace((OLD / 'campaign.toml').read_text(),
                           f'campaign_id = "{m["campaign_id"]}"', f'campaign_id = "{new_id}"')
    text = checked_replace(text, f'archive_root = "{OLD}"', f'archive_root = "{NEW}"')
    manifest.write_text(text)
    (NEW / 'campaign.toml.sha256').write_text(sha(manifest) + '  campaign.toml\n')
    env = {}
    for line in (OLD / 'locked.env').read_text().splitlines():
        assert line.startswith('export ')
        key, value = line[7:].split('=', 1)
        env[key] = shlex.split(value)[0] if value else ''
    old_env = dict(env)
    env.update(GOM_PRODUCTION_CAMPAIGN_ID=new_id, GOM_PRODUCTION_ARCHIVE_ROOT=str(NEW),
               GOM_PRODUCTION_MANIFEST_PATH=str(manifest),
               GOM_PRODUCTION_MANIFEST_COMPANION_PATH=str(NEW / 'campaign.toml.sha256'),
               GOM_PRODUCTION_MANIFEST_SHA256=sha(manifest))
    (NEW / 'locked.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k, v in env.items()))
    check = tomllib.loads(manifest.read_text())
    assert {k: v for k, v in check.items() if k not in ('campaign_id', 'archive_root')} == {
        k: v for k, v in m.items() if k not in ('campaign_id', 'archive_root')}
    changed_source = sorted(p.name for p in (NEW / 'source').iterdir()
                            if not (OLD / 'source' / p.name).exists() or sha(p) != sha(OLD / 'source' / p.name))
    assert set(changed_source) == {'README.md', 'stage_recovery.py', 'pair_input_audit.jl',
        'test_pair_input_audit.jl', 'candidate_guard.sh',
        'gom_step62_effective_pc_global_plateau_preflight.sbatch',
        'gom_step62_effective_pc_global_plateau_full.sbatch'}
    record = dict(passed=True, old_root=str(OLD), recovery_root=str(NEW),
                  fixed_error='Unapproved production/candidate difference: metadata.wrr_contact_angle_deg',
                  specific_sha256=sha(spec), table_sha256=sha(tables), changed_source=changed_source,
                  changed_env=sorted(k for k in env if env[k] != old_env[k]),
                  simulation_physics_and_workflow_identical=True, fresh_restart=True,
                  old_artifacts_preserved=True, upscaling_rerun=False)
    (NEW / 'recovery_contract.json').write_text(json.dumps(record, indent=2) + '\n')
    fs = [p for p in NEW.iterdir() if p.is_file()] + list((NEW / 'source').iterdir())
    (NEW / 'STUDY_SHA256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(NEW)}\n' for p in sorted(fs)))
    for root in (OLD, NEW):
        subprocess.run(['sha256sum', '--check', '--quiet', 'STUDY_SHA256.txt'], cwd=root, check=True)
        subprocess.run(['sha256sum', '--check', '--quiet', 'INPUTS_SHA256.txt'], cwd=root, check=True)
    for p in (NEW / 'source').iterdir():
        if p.suffix in ('.sh', '.sbatch'):
            subprocess.run(['bash', '-n', str(p)], check=True)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
