"""Collect small validation records and scheduler state, never dense reservoir states."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import subprocess
import tomllib

root=Path(__file__).resolve().parent
rows=json.loads((root/'submissions.json').read_text())
evidence=root/'evidence'
evidence.mkdir(exist_ok=True)
records=[]
for row in rows:
    pf=Path(row['root'])/'results'/('gom_step62_effective_pc_global_plateau_preflight_job'+row['preflight_job'])/row['case']
    record={k:row[k] for k in ['name','case','task','preflight_job','full_job']}
    record['preflight_pass']=(pf/'PASS').is_file()
    if record['preflight_pass']:
        out=evidence/row['name'];out.mkdir(exist_ok=True)
        for name in ['preflight_summary.txt','fault_hysteresis_validation.toml','pair_input_audit.toml','exit_status.txt']:
            if (pf/name).exists(): shutil.copy2(pf/name,out/name)
        validation=tomllib.loads((pf/'fault_hysteresis_validation.toml').read_text())
        assert validation['passed'] and validation['maximum_value_error']==0.0 and validation['maximum_slope_error']==0.0
        assert validation['fault_cells']==32190 and validation['host_cells']==2132892
        record['validation']=validation
    records.append(record)
jobs=','.join(row['full_job'] for row in rows)
scheduler=subprocess.check_output(['squeue','-h','-j',jobs,'-o','%i|%T|%M|%R'],text=True)
(evidence/'full_scheduler.txt').write_text(scheduler)
result={'utc':datetime.now(timezone.utc).isoformat(),'preflight_pass_count':sum(r['preflight_pass'] for r in records),'cases':records}
(evidence/'validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('Preflights passed:',result['preflight_pass_count'],'/ 15')
print('All passing cases: exactly zero fault-drainage and host-original value/AD-slope mismatch.')
print(scheduler)
