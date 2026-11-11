"""Finite export-only qualification. Production adoption is a separate decision."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
import common as C
from exact_export import POLICY

BASE = Path('/orcd/data/juanes/001/shaowen')
P = BASE/'phase1_pc40_adaptive10_campaign_20261030_v1'
AUDIT = BASE/'phase1_held173_audit_20261110_v1/output'
AUDIT_PASS = 'a3916a3497504a0af2e2d7271de4ee9323f50447f0badce0a452cb2effe2cf7d'
PARENT = BASE/'phase1_round3_20261110_v1'
PARENT_PIN = '2d1202371c3d80eea4e99ef6ac32069c665cc0699b6ace1d728fff2ce87b0b8b'
METHOD = 'ea1919d05292929667a2e33f38accd9cf0d935cdecafc5ba349083960867b832'
PY = BASE/'predict_shaowen/candidates/wrr60_phase1_production_candidate_20260912_v1/reference_integration_v1/integration_v1/environments/export_py310_v1/bin/python'
JULIA = '/orcd/software/community/001/pkg/julia/1.10.4/bin/julia'
PROJECT = BASE/'predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908/corrected_hysteresis_15case_20260911_v1/runtime'
DEPOT = str(BASE/'predict_shaowen/candidates/wrr60_pc100_kr10_campaign_20260916_v1/full_fault_dispatch_v2/depots/production_1104')+':/home/shaowen/.julia'


def scope():
    C.require(C.sha(AUDIT/'PASS.json') == AUDIT_PASS, 'Classification changed')
    passed = C.read(AUDIT/'PASS.json')
    C.require(passed['passed'] and passed['read_only'] and passed['held_cores'] == 173, 'Incomplete classification')
    C.require(C.sha(AUDIT/'SCOPE.json') == passed['scope_sha256'], 'Audit scope changed')
    rows = [r for r in C.read(AUDIT/'SCOPE.json')['cores'] if r['category']=='export_minimum_saturation_spacing']
    C.require(len(rows)==16 and len({r['key'] for r in rows})==16, 'Wrong targeted scope')
    for r in rows:
        c = r['case_id'];n = f"core{r['slot']:03d}"
        C.require(not (P/'results'/c/(n+'.zip')).exists() and not (P/'runs'/c/'done'/(n+'.json')).exists(), 'Already accepted core excluded')
        C.require(not (P/'complete_cases'/(c+'.json')).exists(), 'Completed case excluded')
    return rows


def qualify():
    C.require(C.ROOT == C.CODE/'output', 'Private output required')
    lock = C.verify_source()
    C.require(C.sha(PARENT/'LOCK.json') == PARENT_PIN, 'Parent changed')
    C.require(not (C.ROOT/'QUALIFIED_PASS.json').exists(), 'Qualification already completed')
    for phase, count in [('round2_20261109_v1',125),('round3_20261110_v1',55)]:
        C.require(C.read(P/phase/'solve/STATE.json') == dict(active=[],cursor=count), 'Production active')
    rows = scope()
    protected = {}
    request_rows = []
    originals = {}
    for row in rows:
        w = Path(row['original_work'])
        proof = tomllib.loads((w/'flow/CORE_PASS.toml').read_text())
        record = C.read(w/'input/record.json')
        C.require(proof['passed'] and proof['record_sha256']==row['record_sha256']==C.digest(record) and proof['method']==METHOD, 'Saved flow identity')
        C.require(proof['input'] == tomllib.loads((w/'input/INPUT_PASS.toml').read_text()), 'Saved input proof')
        for name,h in proof['artifacts'].items():C.require(C.sha(w/'flow'/name)==h, 'Saved member checksum')
        for original in [w,Path(row['previous_finish_work'])]:
            for f in original.rglob('*'):
                if f.is_file():protected[str(f)]=C.sha(f)
        target=C.ROOT/'private'/row['key']
        C.require(not target.exists(), 'Private trial already exists; reconcile before retry')
        target.mkdir(parents=True)
        shutil.copytree(w/'input',target/'input');shutil.copytree(w/'flow',target/'flow')
        C.write(target/'RECOVERY_PROVENANCE.json',dict(source=C.sha(C.CODE/'LOCK.json'),policy=POLICY,original_work=str(w),original_flow_receipt_sha256=C.sha(w/'flow/CORE_PASS.toml'),classification_pass_sha256=AUDIT_PASS,new_solves=0,production_credit=False))
        originals[row['key']] = {str(f.relative_to(target)):C.sha(f) for folder in ['input','flow'] for f in (target/folder).rglob('*') if f.is_file()}
        request_rows.append(dict(key=row['key'],work=str(target),record_sha256=row['record_sha256']))
    C.write(C.ROOT/'PROTECTED.json',protected)
    C.write(C.ROOT/'ATTEMPT.json',dict(condition='export_minimum_saturation_spacing',prior_unsuccessful_export_attempts_per_core=2,private_qualification_attempt=1,production_recovery_attempts=0,job=os.environ['SLURM_JOB_ID'],keys=[r['key'] for r in rows]))
    with tempfile.TemporaryDirectory(prefix='p1_exact_spacing_') as tmp:
        science=Path(tmp)/'science';science.mkdir()
        parent_lock=C.read(PARENT/'LOCK.json')
        C.require(C.sha(C.CODE/'science_bundle.zip')==parent_lock['science_bundle_sha256'],'Science bundle')
        with zipfile.ZipFile(C.CODE/'science_bundle.zip') as z:
            C.require(set(z.namelist())==set(parent_lock['science']), 'Science inventory')
            for n,h in parent_lock['science'].items():
                b=z.read(n);C.require(hashlib.sha256(b).hexdigest()==h,'Science member')
                path=science/n;C.require(path.resolve().is_relative_to(science.resolve()),'Unsafe path')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
        for f,h in C.read(PARENT/'EXECUTABLE_PINS.json').items():C.require(C.sha(f)==h,'Executable changed')
        subprocess.run(['/usr/bin/python3.12','-B',str(science/'runtime_check.py')],check=True)
        subprocess.run([str(PY),'-B','-m','unittest','-v','test_exact_export'],cwd=C.CODE,check=True)
        req=C.ROOT/'EXPORT_REQUEST.json'
        C.write(req,dict(science=str(science),method=METHOD,source=C.sha(C.CODE/'LOCK.json'),table_source_sha256=parent_lock['science']['vendor/kernel/build_tables.py'],cases=request_rows))
        subprocess.run([str(PY),'-B',str(C.CODE/'export_saved.py'),str(req)],check=True)
        native=C.ROOT/'NATIVE_REQUEST.toml'
        lines=['science = '+json.dumps(str(science)),'method = '+json.dumps(METHOD)]
        for r in request_rows:lines+=['[[cases]]']+[k+' = '+json.dumps(v) for k,v in r.items()]
        with native.open('x') as f:f.write('\n'.join(lines)+'\n')
        env=dict(os.environ,JULIA_LOAD_PATH='@:@stdlib',JULIA_PKG_OFFLINE='true',JULIA_PKG_PRECOMPILE_AUTO='0',JULIA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',JULIA_DEPOT_PATH=str(Path(tmp)/'depot')+':'+DEPOT)
        subprocess.run([JULIA,'--startup-file=no','--threads=1','--project='+str(PROJECT),str(C.CODE/'native_packet.jl'),str(native)],env=env,check=True)
    accepted=[]
    for r in request_rows:
        w=Path(r['work']);native=tomllib.loads((w/'NATIVE_PASS.toml').read_text());ex=C.read(w/'export/EXPORT_PASS.json')
        C.require(native['passed'] and native['method']==METHOD and native['record_sha256']==r['record_sha256'],'Native validation')
        C.require(C.sha(w/'export/sgof.csv')==ex['sha256']==native['sgof_sha256'],'Native table identity')
        C.require(ex['raw_curve_values_unchanged'] and ex['all_distinct_positive_saturation_nodes_preserved'],'Exact-node preservation')
        for name,h in originals[r['key']].items():C.require(C.sha(w/name)==h,'Private raw result changed')
        accepted.append(dict(key=r['key'],work=str(w),sgof_sha256=ex['sha256'],native_sha256=C.sha(w/'NATIVE_PASS.toml'),minimum_spacing=ex['minimum_positive_spacing'],max_native_errors=native['max_errors']))
    for f,h in protected.items():C.require(C.sha(f)==h,'Original saved evidence changed')
    scope()
    C.verify_source()
    C.write(C.ROOT/'QUALIFIED_PASS.json',dict(passed=True,source=C.sha(C.CODE/'LOCK.json'),classification_pass_sha256=AUDIT_PASS,policy=POLICY,tests=8,cores=len(accepted),results=accepted,original_files_unchanged=len(protected),raw_values_unchanged=True,all_existing_numerical_checks_unchanged=True,minimum_spacing_policy_changed=True,production_modified=False,new_solves=0,job=os.environ['SLURM_JOB_ID']))
    print('EXPORT_ONLY_QUALIFIED',len(accepted),'NO_PRODUCTION_CREDIT',flush=True)


if __name__ == '__main__':
    C.require('SLURM_JOB_ID' in os.environ, 'Compute allocation required')
    C.require(sys.argv[1:] == ['qualify'], 'Only isolated qualification implemented')
    qualify()
