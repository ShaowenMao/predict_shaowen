"""Four finite qualification stages; never controls production or launches work."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import zipfile
import common as C
import planning
import slurm as S
import worker
from cache_contract import verify_cache_evidence

LABELS={stage:'cache_packet_'+stage for stage in ('unit','checkpoint','resume','audit')}

def pilot():
    lock=C.verify_source();cases=C.read(C.CODE/'STAGED.json')['pilot_cases']
    C.require(len(cases)==1 and not lock['full_campaign_authorized'],'Qualification only')
    scope=C.read(C.ROOT/'SCOPE.json')
    C.require(scope==dict(purpose='qualification',cases=cases,packet_count=1),'Wrong pilot scope')
    C.require(not (C.ROOT/'LAUNCH_AUTHORIZATION.json').exists(),'Production authorization forbidden')
    p=planning.checked_plan(C.CODE,C.ROOT,cases[0]);records=p['packets'][0]
    C.require(len(records)==32 and p['allowed_packets']==[0] and p['purpose']=='qualification','Pilot packet bound')
    return lock,cases[0],records

def original_unchanged():
    snapshot=C.read(C.CODE/'PILOT_INPUTS.json');p=Path(snapshot['production_root']);case=snapshot['case_id']
    C.require(case not in C.read(p/'STATE.json')['active'],'Original case active again')
    C.require(C.sha(p/'cases'/case/'TERMINAL.json')==snapshot['terminal_sha256'],'Terminal authority changed')
    C.require(C.sha(p/'plans'/f'{case}.json')==snapshot['plan_sha256'],'Original plan changed')
    live=S.command(['squeue','-u','shaowen','-n','p1q_num_'+case,'-h'])
    C.require(not live,'Original worker active')
    for slot in snapshot['slots']:
        w=p/'runs'/case/'cores'/f'{case}_core{slot:03d}'
        C.require(not (p/'results'/case/f'core{slot:03d}.zip').exists(),'Production result appeared; do not repeat')
        for n in ('SOLVE_STARTED.toml','flow','HOLD.json','POSTFLOW_HOLD.json','FLOW_ERROR.toml','NATIVE_PASS.toml'):
            C.require(not (w/n).exists(),'Production science appeared: '+str(w/n))
        for n,h in snapshot['original_input_files'][str(slot)].items():C.require(C.sha(w/n)==h,'Original prepared input changed')

def predecessor(stage):
    prior={'checkpoint':'unit','resume':'checkpoint','audit':'resume'}[stage]
    ack=C.read(C.ROOT/'submissions'/LABELS[prior]/'ACK.json');job=ack['job']
    target=job if prior=='unit' else job+'_0'
    # Retry only an accounting read, never a numerical or publication operation.
    for attempt in range(12):
        text=S.command(['sacct','-X','-n','-P','-j',target,'--format=JobID%40,State,ExitCode'])
        rows=[line.split('|') for line in text.splitlines() if line.split('|')[0]==target]
        if len(rows)==1 and rows[0][1:3]==['COMPLETED','0:0']:break
        if attempt==11:raise RuntimeError('Predecessor not successfully terminal: '+text)
        time.sleep(5)
    if prior=='unit':
        proof=C.read(C.ROOT/'COMPUTE_UNIT_PASS.json')
        C.require(proof['passed'] and proof['source']==C.sha(C.CODE/'LOCK.json'),'Unit gate missing')

def checkpoint_baseline(records):
    result={}
    for r in records[:2]:
        w=C.ROOT/'runs'/r['case_id']/'cores'/C.key(r)
        worker.flow_proof(w,r,C.read(C.CODE/'LOCK.json')['method_sha256'])
        names=['SOLVE_STARTED.toml','input/record.json','input/SOURCE.json','input/INPUT_PASS.toml',
               'flow/CORE_PASS.toml']+['flow/'+n for n in tomllib.loads((w/'flow/CORE_PASS.toml').read_text())['artifacts']]
        result[str(r['slot'])]={n:C.sha(w/n) for n in names}
    return result

def audit(records,lock,case):
    baseline=C.read(C.ROOT/'CHECKPOINT_BASELINE.json');data=[]
    resume=C.read(C.ROOT/'submissions'/LABELS['resume']/'ACK.json')['job']
    # Array job ids and the actual per-task SLURM_JOB_ID may differ.
    attempts=list((C.ROOT/'runs'/case/'attempts').glob('packet0_*'))
    checkpoint_keys={C.key(r) for r in records[:2]}
    found=[]
    for path in attempts:
        b=C.read(path/'BEGIN.json')['actions']
        if all(b.get(k)=='FINISH' for k in checkpoint_keys):found.append(path)
    C.require(len(found)==1,'Resume evidence must be unique')
    attempt=found[0]
    for k in checkpoint_keys:C.require('FLOW_PASS '+k not in (attempt/'flow.log').read_text(),'Saved calculation was repeated')
    C.require(C.read(attempt/'SUMMARY.json')['verified']==32,'Incomplete resumed packet')
    for r in records:
        path=C.ROOT/'results'/case/f"core{r['slot']:03d}.zip"
        C.verify_archive(path,r,lock['method_sha256'])
        C.require(C.read(C.ROOT/'runs'/case/'done'/f"core{r['slot']:03d}.json")==C.archive_receipt(path,r,lock['method_sha256']),'Archive receipt mismatch')
        with zipfile.ZipFile(path) as z:
            t=lambda n:tomllib.loads(z.read(n).decode())
            j=lambda n:json.loads(z.read(n))
            C.require(j('input/record.json')==r,'Frozen record changed')
            replay=j('input/REPLAY_PASS.json');inp=t('input/INPUT_PASS.toml');flow=t('flow/CORE_PASS.toml');summary=t('flow/summary.toml')
            C.require(replay['passed'] and int(replay['seed'])==int(r['assignment']['exact_replay_seed']) and int(replay['source_row'])==int(r['assignment']['source_library_row']),'Exact input draw changed')
            C.require(inp['sample']==int(r['assignment']['clay_pc_sample_id']) and inp['contact_angle_deg']==40,'Profile changed')
            C.require(flow['passed'] and flow['input']==inp and flow['record_sha256']==C.digest(r),'Flow identity')
            for n,h in flow['artifacts'].items():C.require(hashlib.sha256(z.read('flow/'+n)).hexdigest()==h,'Flow artifact changed')
            verify_cache_evidence(summary,flow,j('input/SOURCE.json'),C.sha(C.CODE/'LOCK.json'),lock['science'])
            C.require(summary['same_mip_microstates'] and summary['pc_points']==100 and 2<=summary['kr_states']<=10,'Scientific state budget')
            C.require(summary['phase_solver_calls']<=20 and summary['intrinsic_solver_calls']==1,'Flow solve budget')
            life=summary['native_lifetime'];C.require(life['numeric_created']==life['numeric_released'] and life['symbolic_created']==life['symbolic_released'],'Native objects retained')
            C.require(len(list(csv.DictReader(io.StringIO(z.read('flow/pc100.csv').decode()))))==100,'Pc coverage')
            native=t('NATIVE_PASS.toml');export=j('export/EXPORT_PASS.json')
            C.require(native['passed'] and export['passed'] and native['record_sha256']==C.digest(r) and native['method']==lock['method_sha256'],'Native identity')
            C.require(hashlib.sha256(z.read('export/sgof.csv')).hexdigest()==native['sgof_sha256']==export['sha256'],'Export changed')
            for n,h in baseline.get(str(r['slot']),{}).items():C.require(hashlib.sha256(z.read(n)).hexdigest()==h,'Checkpoint recalculated or altered')
            data.append(dict(slot=r['slot'],cells=summary['cells'],seconds=summary['wall_seconds'],peak_kib=summary['memory']['VmHWM_kib'],fast_path=summary['fast_path']))
    # A real accepted-packet invocation must be a no-op, not merely a classifier test.
    before={str(r['slot']):C.sha(C.ROOT/'results'/case/f"core{r['slot']:03d}.zip") for r in records}
    worker.run(case,0)
    C.require(not (C.ROOT/'runs'/case/'attempts'/f"packet0_{os.environ['SLURM_JOB_ID']}").exists(),'No-op performed work')
    C.require(before=={str(r['slot']):C.sha(C.ROOT/'results'/case/f"core{r['slot']:03d}.zip") for r in records},'Accepted result changed')
    original_unchanged();C.verify_source()
    C.require(not list((C.ROOT/'complete_cases').glob('*.json')),'Pilot cannot complete full case')
    C.write(C.ROOT/'PACKET_QUALIFIED_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),
        configuration=lock['configuration_sha256'],method=lock['method_sha256'],verified_cores=32,
        checkpoint_results_preserved=2,accepted_packet_noop=True,production_modified=False,
        production_adopted=False,old_campaign_resumed=False,details=data))
    print('PACKET_QUALIFIED_PASS',flush=True)

def main(stage):
    lock,case,records=pilot()
    if stage=='unit':
        subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(C.CODE),'-p','test_*.py','-v'],cwd=C.CODE,check=True)
        C.write(C.ROOT/'COMPUTE_UNIT_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json')));return
    predecessor(stage);original_unchanged()
    if stage=='checkpoint':
        worker.run(case,0,stop_after=2)
        C.write(C.ROOT/'CHECKPOINT_BASELINE.json',checkpoint_baseline(records))
        print('CHECKPOINT_QUALIFICATION_PASS',flush=True)
    elif stage=='resume':
        C.require(checkpoint_baseline(records)==C.read(C.ROOT/'CHECKPOINT_BASELINE.json'),'Saved flow changed before resume')
        worker.run(case,0)
    elif stage=='audit':audit(records,lock,case)
    else:raise ValueError(stage)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=tuple(LABELS));main(parser.parse_args().stage)
