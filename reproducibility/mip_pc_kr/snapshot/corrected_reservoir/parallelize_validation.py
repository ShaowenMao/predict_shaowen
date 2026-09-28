"""Let independent preflights run alongside unit tests; keep both full-run gates."""
from pathlib import Path
import json
import subprocess

root=Path(__file__).resolve().parent
rows=json.loads((root/'submissions.json').read_text())
unit='22588250'
assert len(rows)==15
def run(*args):
    return subprocess.check_output(args,text=True).strip()
for row in rows:
    for key in ['preflight_job','full_job']:
        status=run('squeue','-h','-j',row[key],'-o','%T %j')
        assert status.startswith('PENDING h15_'),(row[key],status)
audit=[]
# Install every stronger full-run gate before releasing any preflight.
for row in rows:
    dep=f'afterok:{unit}:{row["preflight_job"]}'
    subprocess.run(['scontrol','update','JobId='+row['full_job'],'Dependency='+dep],check=True)
    audit.append({'job':row['full_job'],'dependency':dep})
for row in rows:
    subprocess.run(['scontrol','update','JobId='+row['preflight_job'],'Dependency='],check=True)
    audit.append({'job':row['preflight_job'],'dependency':''})
(root/'validation_dependency_update.json').write_text(json.dumps(audit,indent=2)+'\n')
print('15 independent preflights; all 15 full runs require both shared tests and their own preflight.')
