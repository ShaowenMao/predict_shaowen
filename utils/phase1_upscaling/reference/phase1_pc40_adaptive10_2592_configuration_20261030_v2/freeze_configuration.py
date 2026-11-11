"""Freeze a launch-disabled adaptive successor; --verify is strictly read-only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tomllib

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'phase1_pc40_2592_configuration_20261030_v1'
PILOT = ROOT.parent/'phase1_pc40_adaptive10_integration_20261030_v1'
PARENT_PIN = 'b57b9b565ad0c6dce221314dda627ed7153c6b7e35980fe9a510572f9d7553cd'
PILOT_PIN = '7ac936d57d4542059ebd521096585c07925f9dc105c502be0ae5cd4d92a88572'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def write_json(path,obj):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
        f.write('\n')

def copy(src,dst,expected):
    require(sha(src)==expected and not dst.exists(),f'Copy source/target: {src}')
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    require(sha(dst)==expected,f'Copy changed: {dst}')

def parent_routes(root):
    import pandas as pd
    path=root/'provenance/parent_freeze_configuration.py'
    spec=importlib.util.spec_from_file_location('pc40_parent_route_checks',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    config=json.loads((root/'CONFIG.json').read_text())
    profiles={k:dict(json.loads((root/v['file']).read_text()),_sha256=v['sha256'])
              for k,v in config['profiles'].items()}
    cases=pd.read_csv(root/'CASE_ASSIGNMENTS.csv')
    scenarios=pd.read_csv(root/'sampling/phase1/scenario_catalog.csv')
    module.routes(cases,scenarios,profiles)
    rejected=0
    for column,value in [('clay_pc_sample_id',6),('co2_brine_contact_angle_deg',70),
                         ('physics_profile_id','atlas_sample6_ca40'),('physics_profile_sha256','0'*64),
                         ('host_curve_sha256','0'*64),('master_curve_sha256','0'*64),
                         ('array_index_zero_based',1),('master_endpoint_sg',.7),
                         ('output_relative_namespace','old_production/overwrite')]:
        bad=cases.copy();bad.loc[0,column]=value
        try:
            module.routes(bad,scenarios,profiles)
        except (ValueError,KeyError):
            rejected+=1
        else:
            raise ValueError('Bad route accepted: '+column)
    require(rejected==9,'Negative route tests')
    return rejected

def pilot_acceptance():
    require(sha(PILOT/'RELEASE.json')==PILOT_PIN,'Pilot release pin')
    release=json.loads((PILOT/'RELEASE.json').read_text())
    for p,h in release['files'].items():
        require(sha(PILOT/p)==h,'Pilot source changed: '+p)
    passed=json.loads((PILOT/'INTEGRATION_PASS.json').read_text())
    terminal=json.loads((PILOT/'COMPLETION_VERIFIED.json').read_text())
    require(passed['passed'] and passed['runs']==6 and passed['source_release_sha256']==PILOT_PIN,'Integration acceptance')
    require(passed['samples']==[4,6] and passed['contact_angle_deg']==40,'Pilot profiles')
    require(terminal['passed'] and terminal['job_id']=='27293942' and
            terminal['state']=='COMPLETED' and terminal['exit_code']=='0:0','Slurm completion')
    require(not passed['full_campaign_launched'] and not passed['old_campaign_resumed'],'Launch scope')
    native=tomllib.loads((PILOT/'NATIVE_PASS.toml').read_text())
    expected={c['key'] for c in json.loads((PILOT/'cases.json').read_text())}
    require({r['key'] for r in passed['records']}==expected and len(expected)==6,'Pilot coverage')
    for key in expected:
        r=PILOT/'results'/key
        receipt=tomllib.loads((r/'CORE_PASS.toml').read_text())
        require(receipt['passed'] and receipt['source_release_sha256']==PILOT_PIN,'Core receipt')
        for p,h in receipt['artifacts'].items():
            require(sha(r/p)==h,'Core artifact changed')
        require(sha(r/'export/sgof.csv')==next(x['sha256'] for x in native['records'] if x.get('key')==key),'Native import identity')
    return release,passed,terminal

def build():
    require(not (ROOT/'LOCK.json').exists() and not (ROOT/'CONFIG.json').exists(),'Already built')
    require(sha(PARENT/'LOCK.json')==PARENT_PIN,'Parent lock pin')
    oldlock=json.loads((PARENT/'LOCK.json').read_text())
    release,passed,terminal=pilot_acceptance()
    replaced={'README.md','CONFIG.json','VERIFY.json','freeze_configuration.py'}
    for p,h in oldlock['files'].items():
        require(sha(PARENT/p)==h,'Parent changed: '+p)
        if p in replaced:
            name='parent_freeze_configuration.py' if p=='freeze_configuration.py' else 'parent_'+p
            copy(PARENT/p,ROOT/'provenance'/name,h)
        else:
            copy(PARENT/p,ROOT/p,h)
    copy(PARENT/'LOCK.json',ROOT/'provenance/parent_LOCK.json',PARENT_PIN)
    for p,h in release['files'].items():
        copy(PILOT/p,ROOT/'integration_source'/p,h)
    copy(PILOT/'RELEASE.json',ROOT/'integration_source/RELEASE.json',PILOT_PIN)
    for name in ('AdaptiveSelector.jl','AdaptiveCurve.jl','test_selector.jl','fixtures.toml'):
        copy(PILOT/name,ROOT/'adaptive'/name,release['files'][name])
    for name in ('INTEGRATION_PASS.json','NUMERICAL_PASS.toml','EXPORT_PASS.json','NATIVE_PASS.toml',
                 'RUNTIME_PASS.json','COMPLETION_VERIFIED.json','ALLOCATION_VERIFIED.json',
                 'SUBMISSION_INTENT.json','LOCAL_PIN_PASS.json','accounting.txt',
                 'collect_completion.py','RESULTS.md','pilot_results_27293942.tar.gz'):
        copy(PILOT/name,ROOT/'evidence/adaptive_integration'/name,sha(PILOT/name))
    config=json.loads((PARENT/'CONFIG.json').read_text())
    oldnum=config['numerics']
    config.update(schema_version=2,freeze_id=ROOT.name,created_utc=datetime.now(timezone.utc).isoformat(),
                  parent_configuration=PARENT.name,parent_lock_sha256=PARENT_PIN,
                  state='FROZEN_INTEGRATION_QUALIFIED_NOT_LAUNCHED')
    config.pop('upscaling_jobs_submitted',None)
    config['production_upscaling_jobs_submitted']=0
    config['isolated_integration_jobs_submitted']=1
    runtime=dict(oldnum['runtime_pins'])
    runtime['kernel_fixed_selector_dependency_sha256']=runtime.pop('selector_sha256')
    selector={'id':'adaptive10_endpoint_delta_v1','source':'adaptive/AdaptiveSelector.jl',
              'sha256':release['files']['AdaptiveSelector.jl'],
              'curve_source':'adaptive/AdaptiveCurve.jl','curve_sha256':release['files']['AdaptiveCurve.jl'],
              'budget':10,'shared_gas_brine_states':True,
              'endpoints':'first and final Pc states always queried',
              'criterion':'largest max(abs(delta Krw), abs(delta Krg)) between queried interval endpoints',
              'tie_breaks':['widest Sg interval','earliest eligible index'],
              'next_query':'midpoint rank among unique-Sg eligible state indices',
              'plateau_policy':'retain first and final states; fewer than 10 states if fewer eligible',
              'replay':'exact drainage history from initial state per query; hash each visited microstate',
              'dense_Kr_reference_access_during_selection':False,
              'accuracy_claim':'tested adaptive heuristic, not uniformly superior for every core or phase'}
    config['numerics'].update(pilot_source='integration_source',qualified_job_id='27293942',
                             previous_fixed_pilot_job_id='27286397',runtime_pins=runtime,kr_selector=selector)
    config['execution_gate']['before_launch']=[
        'explicit full-campaign launch authorization',
        'adapt and qualify campaign fine-core generation for frozen stochastic assignments and both profiles',
        'qualify 32-core worker checkpoints, recovery and compact-output lifecycle',
        'qualify continuous eight-case scheduling and actual Slurm allocations',
        'population/largest-grid and packet memory checks; retain separate high-memory exceptions']
    config['planned_execution']={'implementation_status':'not_deployed_or_qualified_by_this_pilot',
        'cores_per_worker':32,'packets_per_522_core_case':17,'continuous_case_window':8,
        'array_throttle':None,'scheduler':'Slurm','old_campaign_resumed':False}
    method={'profiles':config['profiles'],'settings':config['explicit_local_cell_settings'],
            'numerics':{k:config['numerics'][k] for k in ('pc_states','max_kr_states','flow_axis','solver','kr_selector')},
            'pipeline_hash':runtime['pipeline_hash'],'bridge_sha256':runtime['bridge_sha256'],
            'pressure_plan_sha256':oldlock['files']['physics/vendor/PressurePlan.jl'],
            'export_sha256':oldlock['files']['physics/vendor/kernel/build_tables.py']}
    write_json(ROOT/'METHOD.json',method)
    config['method_manifest']={'file':'METHOD.json','sha256':sha(ROOT/'METHOD.json')}
    write_json(ROOT/'CONFIG.json',config)
    rejected=parent_routes(ROOT)
    unchanged={p:h for p,h in oldlock['files'].items() if p.startswith(('sampling/','profiles/','reference_tables/'))
               or p in ('CASE_ASSIGNMENTS.csv','SAMPLING_AUDIT.json','EXTERNAL_LIBRARY_LOCK.json')}
    require(all(sha(ROOT/p)==h for p,h in unchanged.items()),'Assignments/profiles changed')
    write_json(ROOT/'VERIFY.json',{
        'passed':True,'all_case_routes_verified':2592,'negative_routing_tests_passed':rejected,
        'unchanged_parent_assignment_count':1353024,'unchanged_parent_files':unchanged,
        'sampling_redrawn':False,'parent_lock_sha256':PARENT_PIN,'parent_package_unchanged':True,
        'external_library_contents_rehashed_this_version':False,
        'inherited_sampling_audit':'SAMPLING_AUDIT.json',
        'integration':passed,'slurm_completion':terminal,
        'campaign_launched':False,'old_campaign_resumed':False,
        'qualification_scope':'six saved geometries/profile combinations, scientific configuration and native table import; not campaign deployment'})
    files={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(ROOT.rglob('*'))
           if p.is_file() and '__pycache__' not in p.parts}
    write_json(ROOT/'LOCK.json',{'freeze_id':ROOT.name,'launch_authorized':False,'files':files})
    with (ROOT/'LOCK.sha256').open('x') as f:
        f.write(sha(ROOT/'LOCK.json')+'  LOCK.json\n')
    print('FROZEN',len(files),'files; no production campaign launched')

def verify():
    lock=json.loads((ROOT/'LOCK.json').read_text())
    require(sha(ROOT/'LOCK.json')==(ROOT/'LOCK.sha256').read_text().split()[0],'Lock digest')
    require(not lock['launch_authorized'],'Lock launch gate')
    for p,h in lock['files'].items():
        require(sha(ROOT/p)==h,'Frozen file changed: '+p)
    config=json.loads((ROOT/'CONFIG.json').read_text())
    require(not config['launch_authorized'] and not config['execution_gate']['campaign_launch_authorized'],'Launch gate')
    require(config['production_upscaling_jobs_submitted']==0,'Production scope')
    sel=config['numerics']['kr_selector']
    require(sel['id']=='adaptive10_endpoint_delta_v1' and sel['budget']==10,'Adaptive policy')
    require(sha(ROOT/sel['source'])==sel['sha256'] and sha(ROOT/sel['curve_source'])==sel['curve_sha256'],'Adaptive identity')
    require(sha(ROOT/config['method_manifest']['file'])==config['method_manifest']['sha256'],'Method identity')
    v=json.loads((ROOT/'VERIFY.json').read_text())
    require(v['passed'] and v['integration']['passed'] and v['integration']['runs']==6,'Qualification')
    require(all(sha(ROOT/p)==h for p,h in v['unchanged_parent_files'].items()),'Parent assignment identity')
    parent_routes(ROOT)
    print('FREEZE_VERIFY_PASS',len(lock['files']),'files / 2592 unchanged cases; adaptive10 / samples 4,6 / 40 degrees; launch disabled')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--build',action='store_true')
    group.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.build:
        build()
    verify()
