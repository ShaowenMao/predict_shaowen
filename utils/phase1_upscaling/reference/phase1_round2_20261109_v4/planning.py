"""Case-local frozen assignment lookup; no old ledgers or result census."""
import csv
import gzip
from pathlib import Path
from common import require,read,sha,digest,key,write

PACKET=32
WINDOW=8

def packets(records):
    records=sorted(records,key=lambda r:r['slot'])
    require(len({r['case_id'] for r in records})==1,'Mixed case packet plan')
    require(len({key(r) for r in records})==len(records),'Duplicate core')
    return [records[i:i+PACKET] for i in range(0,len(records),PACKET)]

def case_records(configuration,case_id):
    f=Path(configuration);lock=read(f/'LOCK.json')
    require(sha(f/'CASE_ASSIGNMENTS.csv')==lock['files']['CASE_ASSIGNMENTS.csv'],'Case catalog changed')
    with (f/'CASE_ASSIGNMENTS.csv').open() as s:matches=[r for r in csv.DictReader(s) if r['case_id']==case_id]
    require(len(matches)==1,'Unknown case')
    case=matches[0]
    path='sampling/phase1/manifests/'+case['scenario_id']+'.csv.gz'
    require(sha(f/path)==lock['files'][path],'Scenario assignments changed')
    with gzip.open(f/path,'rt') as stream:rows=[r for r in csv.DictReader(stream) if r['case_id']==case_id]
    require(len(rows)==522,'Incomplete case')
    result=[]
    for r in rows:
        require(r['scenario_id']==case['scenario_id'] and int(r['clay_pc_sample_id'])==int(case['clay_pc_sample_id']) and int(r['co2_brine_contact_angle_deg'])==40,'Profile routing')
        require(int(r['predict_seed'])==int(r['exact_replay_seed']) and 0<int(r['predict_seed'])<2**32,'Exact replay seed')
        require(int(r['predict_realization_id'])==int(r['source_library_row']),'Source row convention')
        slot=87*(int(r['window_id'][4:])-1)+int(r['slice_id'])
        result.append(dict(case_id=case_id,slot=slot,profile=case['physics_profile_id'],
            profile_sha256=case['physics_profile_sha256'],assignment=r,configuration_sha256=sha(f/'LOCK.json')))
    require({r['slot'] for r in result}==set(range(1,523)),'Missing/repeated positions')
    return sorted(result,key=lambda r:r['slot'])

def advance(state,queue,terminal,launch,save):
    require(0<=state['cursor']<=len(queue),'Invalid cursor')
    require(len(set(state['active']))==len(state['active'])<=WINDOW,'Invalid active cases')
    state['active']=[c for c in state['active'] if not terminal(c)]
    save(state)
    for c in state['active']:launch(c)
    while len(state['active'])<WINDOW and state['cursor']<len(queue):
        c=queue[state['cursor']];state['cursor']+=1
        if terminal(c):save(state);continue
        state['active'].append(c)
        save(state)
        launch(c)
    return state

def prepare_plan(code,root,case_id):
    """One bounded scenario read at admission, never a completed-result scan."""
    path=root/'plans'/f'{case_id}.json';seal=path.with_suffix('.seal.json')
    if path.exists() and (seal.exists() or case_id in read(code/'LOCK.json')['plans']):
        return checked_plan(code,root,case_id)
    records=case_records(code/'configuration',case_id)
    with (code/'configuration/sampling/phase1/scenario_catalog.csv').open() as f:
        scenarios={r['scenario_id']:r for r in csv.DictReader(f)}
    with (code/'replay_metadata.csv').open() as f:
        metadata={(r['ScenarioLabel'],r['Window'],r['CaseLabel']):r for r in csv.DictReader(f)}
    for r in records:
        a=r['assignment'];s=scenarios[a['scenario_id']];m=metadata[s['scenario_label'],a['window_id'],s['case_label']]
        r['geology']=s;r['source_metadata']={k:m[k] for k in ('SeedBase','NumAttempts','NumRejected')}
    plan=dict(case_id=case_id,records=records,packets=packets(records),purpose='production')
    write(path,plan)
    write(seal,dict(source=sha(code/'LOCK.json'),plan_sha256=sha(path)))
    return plan

def checked_plan(code,root,case_id):
    path=root/'plans'/f'{case_id}.json';lock=read(code/'LOCK.json')
    if path.with_suffix('.seal.json').exists():
        seal=read(path.with_suffix('.seal.json'))
        require(seal['source']==sha(code/'LOCK.json') and seal['plan_sha256']==sha(path),'Case plan seal changed')
    else:
        require(case_id in lock['plans'] and sha(path)==lock['plans'][case_id],'Unknown/unsealed case plan')
    return read(path)
