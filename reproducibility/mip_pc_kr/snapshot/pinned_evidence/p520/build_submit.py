from pathlib import Path
import json,hashlib,subprocess,tarfile,shlex
HERE=Path(__file__).resolve().parent
PARENT='/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/wrr60_phase1_production_candidate_20260912_v1/broader_numerical_v1'
ROOT=PARENT+'/core518_managed_tight22_v1';BASE=PARENT+'/releases/be634618c2500c8a'
REF=PARENT+'/core518_recovery_v1/results/job22737254'
assert not (HERE/'submitted_verified.json').exists()
code="from pathlib import Path;import json,hashlib;p=Path("+repr(REF)+");files=[p/'baseline/completion.toml',p/'baseline_memory.json',p/'baseline.toml']+list((p/'planned60').rglob('*'));print(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files if f.is_file()}))"
inputs=json.loads(subprocess.check_output(['ssh','engaging','/usr/bin/python3.12 -c '+shlex.quote(code)],text=True,timeout=120))
assert REF+'/baseline_memory.json' in inputs
files={n:(HERE/n).read_bytes().replace(b'\r\n',b'\n') for n in ('run.py','DESIGN.md','build_submit.py')}
job=(HERE.parent/'core518_recovery_v1/job.sh').read_text().replace('"$PY" -u -B -m unittest -v test_recovery\n','').replace('recover.py','run.py')
job=job.replace('exec "$PY"', '"$PY" -B -m unittest discover -s '+BASE+' -p test_checks.py -v\nexec "$PY"')
files['job.sh']=job.replace('\r\n','\n').encode()
files['spec.json']=(json.dumps(dict(root=ROOT,base=BASE,reference=REF,inputs=inputs),sort_keys=True,indent=2)+'\n').encode()
lock=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),sort_keys=True,indent=2)+'\n').encode()
ident=hashlib.sha256(lock).hexdigest()[:16];out=HERE/'releases'/ident;out.mkdir(parents=True,exist_ok=False)
for n,b in files.items():(out/n).write_bytes(b)
(out/'source_lock.json').write_bytes(lock)
archive=HERE/(ident+'.tar.gz')
with tarfile.open(archive,'w:gz') as f:f.add(out,arcname=ident)
subprocess.run(['ssh','engaging','mkdir -p '+ROOT+'/releases '+ROOT+'/logs'],check=True)
subprocess.run(['scp',str(archive),'engaging:'+ROOT+'/releases/'],check=True)
subprocess.run(['ssh','engaging','tar -xzf '+ROOT+'/releases/'+archive.name+' -C '+ROOT+'/releases'],check=True)
code='''from pathlib import Path
import subprocess,json
r=Path(ROOT);release=r/'releases'/IDENT
intent=r/'submission.json';assert not intent.exists();intent.write_text(json.dumps(dict(source=IDENT,status='INTENT')))
job=subprocess.check_output(['sbatch','--parsable','--account=mit_amf_advanced_cpu','--qos=mit_amf_advanced_cpu','--partition=mit_normal','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=22G','--time=01:00:00','--no-requeue','--job-name=w60_518_tight22','--output='+str(r/'logs/%j.log'),str(release/'job.sh'),str(release)],text=True,timeout=60).strip().split(';')[0]
assert job.isdigit()
actual=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=45)
v=dict(x.split('=',1) for x in actual.split() if '=' in x)
for k,x in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',MinMemoryNode='22G',Requeue='0').items():assert v[k]==x
d=dict(job=job,actual=actual,root=ROOT,source=IDENT);(r/'submitted_verified.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
'''.replace('ROOT',repr(ROOT)).replace('IDENT',repr(ident))
raw=subprocess.check_output(['ssh','engaging','/usr/bin/python3.12 -c '+shlex.quote(code)],text=True,timeout=120)
(HERE/'submitted_verified.json').write_text(json.dumps(json.loads(raw),indent=2)+'\n');print(raw)
