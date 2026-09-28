"""Freeze isolated campaign orchestration; scientific files copied byte-for-byte."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
HERE=Path(__file__).resolve().parent;P=HERE.parent
def blob(x):return (json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in ('launch.py','job.sh','test_launch.py','build.py')}
parent=P/'broader_replay_v1/releases/c29e94d3d4257bb2'
old=json.loads((parent/'source_lock.json').read_bytes())
for n,h in old['files'].items():
    if n.startswith('frozen/') or n in ('runtime_inventory.json','verify_matlab_startup.m','temp_paths.py'):
        b=(parent/n).read_bytes();assert sha(b)==h;files[n]=b
for n in ('features.m','tensor_descriptors.m'):
    files[n]=(P/'broader_replay_v1'/n).read_bytes()
context=P/'qualification_dispatch_v1/releases/c3fcc4f374206f66'
for n in ('assignment_plan.py','dispatch_checks.py','policy.json','SELECTION.json','test_dispatch_checks.py'):
    files[n]=(context/n).read_bytes()
spec=json.loads((context/'spec.json').read_text())
spec.update(root='/orcd/data/juanes/001/shaowen/predict_shaowen/campaigns/wrr60_phase1_2430_20260914_v1',
    campaign_authorized=True,authorization='User requested campaign batches on 2026-09-14; preserve original production.',
    stage='exact_draw_replay_wave01',numerical_dispatch_approved=False,
    maintenance_start='2026-09-15T00:00:00',maintenance_end='2026-09-15T21:00:00',
    cpu_ceiling=500,memory_ceiling_gib=1984,
    matlab_version=json.loads((parent/'spec.json').read_text())['matlab_version'],
    campaign_manifest_sha256='adf73c15b1c4edc4b4721c7666a2c2b56aaf800088fff3599a5541dce0ed3dde',
    scope_note='First-wave replay only. Same scientific sources; fresh process per core. Unknown >150k-cell geometry and all unqualified numerical classes held.')
import tomllib
manifest=(P/'case_design_review_v1/campaign.toml').read_bytes();assert sha(manifest)==spec['campaign_manifest_sha256']
files['campaign.json']=blob(tomllib.loads(manifest.decode()))
files['spec.json']=blob(spec)
lock=blob(dict(files={n:sha(b) for n,b in sorted(files.items())}));ident=sha(lock)[:16]
out=HERE/'releases'/ident;out.mkdir(parents=True,exist_ok=False)
for n,b in files.items():
    p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
(out/'source_lock.json').write_bytes(blob(dict(release=ident,files={n:sha(b) for n,b in sorted(files.items())})))
subprocess.run([sys.executable,'-B','-m','unittest','-v','test_launch','test_dispatch_checks'],cwd=out,check=True)
with tarfile.open(HERE/(ident+'.tar.gz'),'w:gz') as t:t.add(out,arcname=ident)
result=dict(release=ident,root=spec['root'],source_lock=sha((out/'source_lock.json').read_bytes()),archive_sha256=sha((HERE/(ident+'.tar.gz')).read_bytes()))
(HERE/'release.json').write_bytes(blob(result));print(json.dumps(result))
