"""Change only the 15 pending study jobs' wall limit, preserving their job ids."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess

root=Path(__file__).resolve().parent
rows=json.loads((root/'submissions.json').read_text())
assert len(rows)==15
evidence=[]
for row in rows:
    job=row['full_job']
    status=subprocess.check_output(['squeue','-h','-j',job,'-o','%T %j'],text=True).strip()
    assert status.startswith('PENDING h15_'),(job,status)
    info=subprocess.check_output(['scontrol','show','job',job],text=True)
    assert 'TimeLimit=4-00:00:00' in info
    evidence.append({'job':job,'before':info})
for entry in evidence:
    subprocess.run(['scontrol','update','JobId='+entry['job'],'TimeLimit=3-00:00:00'],check=True)
    entry['after']=subprocess.check_output(['scontrol','show','job',entry['job']],text=True)
    assert 'TimeLimit=3-00:00:00' in entry['after']
(root/'maintenance_time_limit_adjustment.json').write_text(json.dumps({
    'utc':datetime.now(timezone.utc).isoformat(),
    'reason':'Four days crosses scheduled Sep 15 maintenance; three days fits the current start window.',
    'solver_settings_changed':False,'jobs':evidence},indent=2)+'\n')
print('Updated only the 15 pending full-study jobs to 72 hours.')
