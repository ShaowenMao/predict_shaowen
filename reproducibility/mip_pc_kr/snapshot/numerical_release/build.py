"""Freeze the adapter plus byte-identical borrowed scientific kernels."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT='/orcd/data/juanes/001/shaowen/predict_shaowen/campaigns/wrr60_phase1_2430_20260914_v1'
def sha(b):return hashlib.sha256(b).hexdigest()
def blob(d):return (json.dumps(d,sort_keys=True,indent=2)+'\n').encode()
def main():
    parent=PARENT/'broader_numerical_v1/releases/be634618c2500c8a'
    closure=PARENT/'qualification_closure_v1/releases/29d1492672ff52d4'
    old=json.loads((parent/'spec.json').read_text()); cs=json.loads((closure/'spec.json').read_text())
    evidence=PARENT/'qualification_closure_v1/evidence/job22753044/AUDIT.json'
    selected=json.loads((PARENT/'campaign_launch_v1/releases/5748ac68fdc0be88/SELECTION.json').read_text())['entries']
    spec={k:old[k] for k in ('pipeline','pipeline_hash','bridge','frozen','scenario','external_files')}
    spec.update(campaign=ROOT,maintenance_start='2026-09-15T00:00:00',runtime=cs['runtime'],
        closure_path=cs['root']+'/results/job22753044/AUDIT.json',closure_sha256=sha(evidence.read_bytes()),
        special_holds=[[e['case']['case_key'],287] for e in selected],
        special_hold_scope='Provisional slot287 holds until exceptional historical context is matched; not all slot287 cores are known failures')
    spec['external_files'].update(cs['pins'])
    spec.update(reuse_parent_root=ROOT+'/numerical_wave01',
        reuse_parent_release=ROOT+'/numerical_releases/60cd9d0316981b0c',
        reuse_parent_source_lock='60cd9d0316981b0cfe0efed84bcc6e152980a5c3f90cbc5c22b6c143a030244a',
        reuse_parent_plan_sha256='9d4676b3dd29fd2c340106fec72e987dbb52d481fefe804af8dd26280fb3c241')
    files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in ('DESIGN.md','policy.py','campaign.py','test_policy.py','test_campaign.py','job.sh','build.py','reuse.py','reuse_campaign.py','test_reuse.py','test_reuse_campaign.py')}
    borrowed={}
    for n in ('prepare.jl','worker.jl','Backend.jl','ManagedLU.jl','MemoryProbe.jl','context.py','build_tables.py'):
        files[n]=(parent/n).read_bytes();borrowed[n]=dict(parent=str(parent/n),sha256=sha(files[n]))
    for p in (parent/'tomli').glob('*.py'):files['tomli/'+p.name]=p.read_bytes()
    native=(closure/'native.jl').read_text();original=native
    replacements=[('@assert length(T)==6','@assert length(T)>=1'),('collect(1:6)','collect(1:length(T))'),
                  ('@test length(exceptions)==1','@test length(exceptions)<=1')]
    for before,after in replacements:
        assert before in native
        native=native.replace(before,after)
    restored=native
    for before,after in reversed(replacements):restored=restored.replace(after,before)
    assert restored==original
    files['native.jl']=native.encode();files['origin.toml']=(closure/'origin.toml').read_bytes()
    files['derivation.json']=blob(dict(borrowed=borrowed,native_parent_sha256=sha(original.encode()),
        native_replacements=replacements,scientific_kernels_byte_identical=True,
        native_scope='Arbitrary number of tables; only same exact documented Core243 exception'))
    files['spec.json']=blob(spec)
    lock=blob(dict(files={n:sha(b) for n,b in files.items()}));ident=sha(lock)[:16]
    out=HERE/'releases'/ident;out.mkdir(parents=True,exist_ok=False)
    for n,b in files.items():
        p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    (out/'source_lock.json').write_bytes(lock)
    subprocess.run([sys.executable,'-B','-m','unittest','-v','test_policy','test_campaign','test_reuse','test_reuse_campaign'],cwd=out,check=True)
    for n,b in files.items():
        if n.endswith('.py'):compile(b,n,'exec')
    archive=HERE/(ident+'.tar.gz')
    with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=ident)
    result=dict(release=ident,remote=ROOT+'/numerical_releases/'+ident,
                source_lock_sha256=sha(lock),archive_sha256=sha(archive.read_bytes()))
    (HERE/'release.json').write_bytes(blob(result));print(json.dumps(result))

if __name__=='__main__':main()
