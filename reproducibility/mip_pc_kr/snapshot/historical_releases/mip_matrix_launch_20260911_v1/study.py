"""Isolated, dependency-gated launch of the ten missing MIP reservoir variants."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

SOURCE = Path(__file__).resolve().parent
BASE = Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
COMPARISON = BASE / 'comparative_3case_wrr30_60_pc_conventions_20260910_v1'
OUT = COMPARISON / 'reservoir_matrix_launch_20260911_v1'
TEMPLATE = BASE / 'hybrid_pc60_production_kr_s03_c022_case12_20260911_v1/reservoir_preflightfix1'
JUTUL = Path('/home/shaowen/orcd/scratch/gom_grid/code/immutable/JutulDarcy_ed81af8e')
CAMPAIGN = Path('/orcd/data/juanes/001/shaowen/gom_full_production/campaign_manifests/step62_phase1_2430_independent_full_fault_v1_effective_globalplateau_qoi4_coordfix_pressurefix_20260814/campaign.toml')
CAMPAIGN_SHA = 'adf73c15b1c4edc4b4721c7666a2c2b56aaf800088fff3599a5541dce0ed3dde'
DATASETS = [('s01_c018_case07',30),('s01_c018_case07',60),('s03_c022_case12',60),
            ('s03_c003_case08',30),('s03_c003_case08',60)]
CONVENTIONS = ('physical_prebreakthrough','zero_before_breakthrough')


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def files(root):
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p != root/'source_lock.json'}


def verify_source():
    assert files(SOURCE)==read(SOURCE/'source_lock.json')['files']


def verify_inputs():
    assert sha(CAMPAIGN)==CAMPAIGN_SHA
    for p,h in read(OUT/'protected_sources.json').items():
        assert sha(p)==h, ('Input/source changed',p)


def load_vendor(name):
    spec=importlib.util.spec_from_file_location(name,SOURCE/'vendor'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def matrix():
    return [dict(case=c,angle=a,pc_convention=pc,variant=f'{c}_wrr{a}_{pc}',dataset=i)
            for i,(c,a) in enumerate(DATASETS) for pc in CONVENTIONS]


def test_matrix():
    vs=matrix()
    assert len(vs)==10 and len({v['variant'] for v in vs})==10
    assert not any(v['case']=='s03_c022_case12' and v['angle']==30 for v in vs)
    assert all(sum(v['case']==c and v['angle']==a for v in vs)==2 for c,a in DATASETS)


def dataset_root(case,angle):
    return OUT/'datasets'/f'{case}_wrr{angle}'


def variant_root(case,angle,pc):
    return OUT/'variants'/f'{case}_wrr{angle}_{pc}'


def stage():
    verify_source();test_matrix()
    assert not OUT.exists()
    assert sha(CAMPAIGN)==CAMPAIGN_SHA
    m=read(COMPARISON/'inputs/manifest.json')
    registry=read(COMPARISON/'study_registry.json')
    expected={x['variant'] for x in registry['variants'] if 'existing_job' not in x}
    assert expected=={x['variant'] for x in matrix()} and len(expected)==10
    keep=[x for x in registry['variants'] if 'existing_job' in x]
    assert {x['existing_job'] for x in keep}=={'22438751_1137','22474505_1137'}
    subprocess.run(['sha256sum','--check','--quiet','STUDY_SHA256.txt'],cwd=TEMPLATE,check=True)
    OUT.mkdir()
    protected={str(CAMPAIGN):CAMPAIGN_SHA,str(COMPARISON/'inputs/manifest.json'):sha(COMPARISON/'inputs/manifest.json')}
    for p in [TEMPLATE/'campaign.toml',TEMPLATE/'locked.env',TEMPLATE/'STUDY_SHA256.txt']+list((TEMPLATE/'source').iterdir()):
        if p.is_file():protected[str(p)]=sha(p)
    for c,a in DATASETS:
        root=dataset_root(c,a);(root/'logs').mkdir(parents=True)
        prod=m['cases'][c]['production']
        assert prod['task'] in (262,848,1137)
        protected[prod['specific_path']]=prod['specific_sha256']
        if c=='s03_c022_case12':
            accepted=BASE/'full_fault_wrr60_s03_c022_case12_20260910_v1/accepted'
            refs=BASE/'full_fault_wrr60_s03_c022_case12_20260910_v1/raw_manifest.json'
            protected[str(refs)]=sha(refs)
        else:
            accepted=COMPARISON/f'datasets/{c}/wrr{a}/accepted'
            done=COMPARISON/f'assemblies/{c}/wrr{a}/complete.json'
            assert read(done)['passed']
            entries=BASE/f'hybrid_low_high_wrr30_60_20260911_v1/{c}/wrr{a}/pc_export/entry_metadata.json'
            protected[str(entries)]=sha(entries)
        assert len(list(accepted.glob('slot*.json')))==522
        protected.update({str(p):sha(p) for p in accepted.glob('slot*.json')})
        for pc in CONVENTIONS:(variant_root(c,a,pc)/'logs').mkdir(parents=True)
    (OUT/'logs').mkdir()
    write(OUT/'protected_sources.json',protected)
    write(OUT/'matrix.json',dict(new_variants=matrix(),preserved_existing=keep,production_modified=False))
    print('STAGED ten new variants; two existing jobs preserved',flush=True)


def csvwrite(p,records,delimiter=','):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),delimiter=delimiter);w.writeheader();w.writerows(records)


def command(folder,args):
    folder.mkdir(parents=True)
    start=time.monotonic()
    print('START',folder,flush=True)
    with (folder/'process.log').open('x') as f:
        rc=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,cwd=folder).returncode
    write(folder/'receipt.json',dict(returncode=rc,seconds=time.monotonic()-start,args=args,job=os.environ['SLURM_JOB_ID']))
    assert rc==0, ('Process failed',folder,rc)


def julia(folder,script,args,standalone=False):
    module='deprecated-modules gcc/12.2.0-x86_64 julia/1.12.6' if standalone else 'community-modules julia/1.10.4'
    cmd=['julia','--startup-file=no']
    if standalone:cmd+=['--math-mode=ieee','--check-bounds=yes']
    else:cmd+=['--compiled-modules=no',f'--project={JUTUL}']
    cmd+=[str(script)]+list(map(str,args))
    setup=f'module purge\nmodule load {module}\nexport JULIA_NUM_THREADS=2 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_CPU_TARGET=generic\n'
    command(folder,['bash','-lc',setup+'exec '+shlex.join(cmd)])


def matlab(folder,expression):
    setup='module purge\nmodule load deprecated-modules gcc/12.2.0-x86_64 matlab/matlab-2025b\n'
    command(folder,['bash','-lc',setup+'exec '+shlex.join(['matlab','-batch',expression])])


def mq(p):
    return "'"+str(p).replace("'","''")+"'"


def export_dataset(index):
    import numpy as np
    import scipy,h5py
    from scipy.io import savemat,loadmat
    assert (np.__version__,scipy.__version__,h5py.__version__)==('2.2.6','1.15.3','3.14.0')
    assert os.environ.get('SLURM_JOB_ID')
    verify_source();verify_inputs()
    test_matrix()
    subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(SOURCE/'vendor'),'-p','test_*.py'],check=True)
    case,angle=DATASETS[index];root=dataset_root(case,angle)
    assert not (root/'READY.json').exists()
    attempt=root/f'attempt_job{os.environ["SLURM_JOB_ID"]}';attempt.mkdir()
    m=read(COMPARISON/'inputs/manifest.json');case_info=m['cases'][case]
    if case=='s03_c022_case12':
        accepted=BASE/'full_fault_wrr60_s03_c022_case12_20260910_v1/accepted'
        refs={int(x['slot']):x for x in read(accepted.parent/'raw_manifest.json')['records']}
        entries={i:float(x['entry_pressure_pa']) for i,x in refs.items()}
    else:
        accepted=COMPARISON/f'datasets/{case}/wrr{angle}/accepted'
        ep=BASE/f'hybrid_low_high_wrr30_60_20260911_v1/{case}/wrr{angle}/pc_export/entry_metadata.json'
        entries={int(x['slot']):float(x['entry_pressure_pa']) for x in read(ep)}
    assert set(entries)==set(range(1,523))
    converter=load_vendor('build_tables');bt_converter=load_vendor('build_pc_only')
    raw_curves={};physical=np.empty((6,87),dtype=object);reports=[];raw_manifest=[];sources={}
    for r in case_info['records']:
        slot=r['slot'];a=r['assignment'];w=(slot-1)//87;s=(slot-1)%87
        assert (a['window'],int(a['slice_index']))==(f'famp{w+1}',s+1)
        rp=accepted/f'slot{slot:03}.json';rc=read(rp)
        assert rc['numerical_checks_passed'] and rc['slot']==slot
        if case!='s03_c022_case12':
            assert rc['case']==case and rc['angle']==angle and rc['calculation_key']==r['calculation_key']
            assert rc['input_manifest_sha256']==sha(COMPARISON/'inputs/manifest.json')
        else:
            rr=refs[slot]
            assert (int(rr['sample']),int(rr['seed']))==(int(a['selected_sample_index']),int(a['exact_replay_seed']))
        ar=Path(rc['attempt']);sources[str(rp)]=sha(rp)
        curve=ar/'numerical/coupled/pc_kr_raw.csv';raw=ar/'numerical/planned/raw.bin'
        for rel in ('numerical/coupled/pc_kr_raw.csv','numerical/coupled/completion.toml'):
            path=ar/rel;assert sha(path)==rc['outputs'][rel];sources[str(path)]=sha(path)
        assert tomllib.loads((ar/'numerical/coupled/completion.toml').read_text())['execution_complete']
        if angle==60:
            ap=ar/'numerical/planned/scenario_audit.toml'
            assert sha(ap)==rc['outputs']['numerical/planned/scenario_audit.toml']
            au=tomllib.loads(ap.read_text())
            assert au['passed'] and au['clay_swi']==.2 and au['clay_nw']==3.5 and au['clay_ng']==2.5
            sources[str(ap)]=sha(ap)
        x=np.loadtxt(curve,delimiter=',',skiprows=1);assert x.shape==(100,8)
        physical[w,s],report=converter.convert(x,entries[slot])
        report.update(slot=slot,window=w+1,slice=s+1);reports.append(report);raw_curves[slot]=x
        raw_manifest.append(dict(slot=slot,path=str(raw),sha256=rc['outputs']['numerical/planned/raw.bin']))
    write(attempt/'source_receipts.json',sources)
    csvwrite(attempt/'entries.csv',[dict(slot=i,entry_pressure_pa=entries[i]) for i in range(1,523)])
    csvwrite(attempt/'raw_manifest.tsv',raw_manifest,delimiter='\t')
    julia(attempt/'breakthrough_process',SOURCE/'extract_breakthrough.jl',
          [attempt/'raw_manifest.tsv',attempt/'entries.csv',attempt/'breakthrough'],standalone=True)
    br=tomllib.loads((attempt/'breakthrough/PASS.toml').read_text());assert br['passed'] and br['cores']==522
    thresholds={int(x['slot']):float(x['breakthrough_pa']) for x in csv.DictReader((attempt/'breakthrough/breakthrough.csv').open())}
    bt=np.empty_like(physical);bt_reports=[]
    for slot,x in raw_curves.items():
        w=(slot-1)//87;s=(slot-1)%87
        bt[w,s],report=bt_converter.convert(physical[w,s],x,thresholds[slot])
        report.update(slot=slot);bt_reports.append(report)
    assert all(np.array_equal(physical[w,s][:,:3],bt[w,s][:,:3]) for w in range(6) for s in range(87))
    built={}
    for pc,tables,recs in [(CONVENTIONS[0],physical,reports),(CONVENTIONS[1],bt,bt_reports)]:
        dest=attempt/pc;dest.mkdir();mat=dest/f'candidate_tables_{case}.mat';spec=dest/f'candidate_specific_{case}.mat'
        savemat(mat,dict(exportTables=tables,tableSlotMap=np.arange(1,523).reshape(6,87),
            entryPressurePa=np.array([t[0,3] for t in tables.flat]).reshape(6,87),
            rawSourceSha256=sha(attempt/'source_receipts.json'),entryMetadataSha256=sha(attempt/'entries.csv'),
            schema='candidate_sgof_native_nodes_v1',caseKey=case,wrrContactAngleDeg=angle,pcConvention=pc,
            productionPromotion=False,validationReceiptRequired=True),do_compression=True)
        reread=loadmat(mat)['exportTables'];assert all(np.array_equal(tables[w,s],reread[w,s]) for w in range(6) for s in range(87))
        audit=dict(cores=522,all_structural_checks_passed=True,angle=angle,case=case,pc_convention=pc,
                   source_receipts_sha256=sha(attempt/'source_receipts.json'),records=recs,
                   continuous_sampling_accuracy_claimed=False,production_promotion=False,
                   mip_sweeps=0,phase_flow_solves=0,pc_conventions_share_exact_kr=True)
        write(dest/'audit.json',audit)
        prod=case_info['production'];assert sha(prod['specific_path'])==prod['specific_sha256']
        expression=f'addpath({mq(SOURCE)});make_specific({mq(prod["specific_path"])},{mq(mat)},{mq(dest/"audit.json")},{mq(spec)},{mq(case)},{angle},{mq(pc)});'
        matlab(dest/'specific_process',expression)
        assert read(str(spec)+'.roundtrip.json')['passed']
        built[pc]=dict(tables=str(mat),specific=str(spec),table_sha256=sha(mat),specific_sha256=sha(spec),audit=str(dest/'audit.json'))
    for pc in CONVENTIONS:make_launcher(case,angle,pc,built,attempt)
    write(root/'READY.json',dict(passed=True,attempt=str(attempt),built=built,cores=522,original_draws_verified=True))
    print('EXPORT_READY',case,angle,flush=True)


def toml_values(d):
    return ''.join(f'{k} = {str(v).lower() if isinstance(v,bool) else json.dumps(v)}\n' for k,v in d.items())


def make_launcher(case,angle,pc,built,attempt):
    root=variant_root(case,angle,pc);assert not (root/'campaign.toml').exists()
    (root/'tmp').mkdir();source=root/'source';source.mkdir()
    for p in (TEMPLATE/'source').iterdir():
        if p.name.startswith('gom_step62_') and p.is_file():shutil.copy2(p,source/p.name)
    for name in ('pair_input_audit.jl','README.md'):shutil.copy2(SOURCE/name,source/name)
    prod=read(COMPARISON/'inputs/manifest.json')['cases'][case]['production'];task=prod['task']
    guard=(SOURCE/'candidate_guard.sh.in').read_text().replace('@ROOT@',str(root)).replace('@TASK@',str(task)).replace('@CASE@',case)
    assert '@ROOT@' not in guard;(source/'candidate_guard.sh').write_text(guard)
    for p in source.glob('*.sbatch'):
        text=p.read_text();assert '#SBATCH --array=1137' in text
        p.write_text(text.replace('#SBATCH --array=1137',f'#SBATCH --array={task}'))
    old=tomllib.loads((TEMPLATE/'campaign.toml').read_text());m=copy.deepcopy(old)
    selected=built[pc];spec=Path(selected['specific']);table=Path(selected['tables'])
    m.update(campaign_id=f'mip_matrix_{case}_wrr{angle}_{pc}_20260911',allowed_original_task=task,archive_root=str(root))
    m['baseline'].update(specific_path=prod['specific_path'],specific_sha256=prod['specific_sha256'])
    m['cases']=[dict(prod,specific_path=str(spec),specific_sha256=sha(spec),specific_bytes=spec.stat().st_size)]
    assert m['workflow']==old['workflow'] and len(m['workflow']['retain_years'])==210
    assert m['restart_reuse'] is False and m['production_promotion'] is False
    text=toml_values({k:v for k,v in m.items() if not isinstance(v,(dict,list))})
    for section in ('baseline','workflow','common'):text+='\n['+section+']\n'+toml_values(m[section])
    text+='\n[[cases]]\n'+toml_values(m['cases'][0]);manifest=root/'campaign.toml';manifest.write_text(text)
    (root/'campaign.toml.sha256').write_text(sha(manifest)+'  campaign.toml\n')
    env={}
    for line in (TEMPLATE/'locked.env').read_text().splitlines():
        k,v=line[7:].split('=',1);env[k]=shlex.split(v)[0] if v else ''
    env.update(GOM_PRODUCTION_CAMPAIGN_ID=m['campaign_id'],GOM_PRODUCTION_ARCHIVE_ROOT=str(root),
        GOM_PRODUCTION_MANIFEST_PATH=str(manifest),GOM_PRODUCTION_MANIFEST_COMPANION_PATH=str(manifest)+'.sha256',
        GOM_PRODUCTION_MANIFEST_SHA256=sha(manifest),GOM_PRODUCTION_TASK=str(task),GOM_PRODUCTION_CASE_KEY=case,
        GOM_PRODUCTION_GEOLOGY_ID=prod['geology_id'],GOM_PRODUCTION_REALIZATION_ID=str(prod['realization_id']),
        GOM_PRODUCTION_GEOLOGY_HASH=prod['geology_hash'],GOM_PRODUCTION_LEVEL3_CASE_NAME=prod['level3_case_name'],
        GOM_PRODUCTION_SPECIFIC_MAT=str(spec),GOM_PRODUCTION_SPECIFIC_SHA256=sha(spec),GOM_PRODUCTION_SPECIFIC_BYTES=str(spec.stat().st_size),
        BASELINE_SPECIFIC_MAT=prod['specific_path'],CANDIDATE_TABLES_MAT=str(table),
        MIP_WRR_ANGLE=str(angle),MIP_PC_CONVENTION=pc,PHYSICAL_TABLES_MAT=built[CONVENTIONS[0]]['tables'])
    (root/'locked.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in env.items()))
    inputs={m['common']['path']:m['common']['sha256'],prod['specific_path']:prod['specific_sha256'],
            str(spec):sha(spec),str(table):sha(table),built[CONVENTIONS[0]]['tables']:built[CONVENTIONS[0]]['table_sha256'],
            str(JUTUL/'Manifest.toml'):m['jutul_manifest_sha256'],str(attempt/'source_receipts.json'):sha(attempt/'source_receipts.json'),
            str(attempt/'breakthrough/PASS.toml'):sha(attempt/'breakthrough/PASS.toml'),str(SOURCE/'source_lock.json'):sha(SOURCE/'source_lock.json')}
    (root/'INPUTS_SHA256.txt').write_text(''.join(f'{h}  {p}\n' for p,h in inputs.items()))
    write(root/'comparison_contract.json',dict(passed=True,case=case,angle=angle,pc_convention=pc,
        only_reservoir_physics_change='fault Pc and MIP-derived dip-parallel Kr',same_draws=True,
        production_promotion=False,kr_method='mip_dip_parallel_full_slice',movie_report_states=210,
        cpus=8,memory_gb=18,source_tables=built,all_nonclosure_physics_settings_preserved=True))
    fs=[p for p in root.iterdir() if p.is_file()]+list(source.iterdir())
    (root/'STUDY_SHA256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in sorted(fs)))
    for p in source.iterdir():
        if p.suffix in ('.sh','.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)


def qualify(root):
    import numpy as np
    from scipy.io import loadmat
    assert os.environ.get('SLURM_JOB_ID')
    verify_source();root=Path(root);assert root.parent==OUT/'variants'
    subprocess.run(['sha256sum','--check','--quiet','STUDY_SHA256.txt'],cwd=root,check=True)
    subprocess.run(['sha256sum','--check','--quiet','INPUTS_SHA256.txt'],cwd=root,check=True)
    c=read(root/'comparison_contract.json');rec=c['source_tables'][c['pc_convention']]
    mat=Path(rec['tables']);spec=Path(rec['specific']);common=tomllib.loads((root/'campaign.toml').read_text())['common']['path']
    assert sha(mat)==rec['table_sha256'] and sha(spec)==rec['specific_sha256']
    q=root/f'qualification_job{os.environ["SLURM_JOB_ID"]}';q.mkdir()
    julia(q/'constitutive_process',SOURCE/'vendor/validate_constitutive.jl',[mat,q/'runtime',common,spec])
    tables=loadmat(mat)['exportTables'];flat=list(tables.flat)
    metrics=[lambda t:t[0,3],lambda t:t[-1,0],lambda t:np.max(np.diff(t[:,3])/np.diff(t[:,0])),
             lambda t:np.max(np.diff(t[:,1])/np.diff(t[:,0])),lambda t:np.max(-np.diff(t[:,2])/np.diff(t[:,0]))]
    slots=[]
    for metric in metrics:
        values=[metric(t) for t in flat]
        for i in (int(np.argmin(values))+1,int(np.argmax(values))+1):
            if i not in slots:slots.append(i)
    for i in (1,88,175,262,349,436,522):
        if i not in slots:slots.append(i)
    slots=slots[:7]
    julia(q/'smoke_process',SOURCE/'vendor/smoke_validation_v3.jl',[mat,q/'runtime',q/'smoke',','.join(map(str,slots))])
    result=tomllib.loads((q/'smoke/PASS.toml').read_text())
    assert result['passed'] and result['smoke_simulations']==14 and result['table_sha256']==sha(mat)
    write(root/'qualified.json',dict(passed=True,table_sha256=sha(mat),specific_sha256=sha(spec),
          native_constitutive_tables=522,smoke_simulations=14,receipt_directory=str(q),slots=slots))
    print('QUALIFIED',root.name,flush=True)


def submit():
    import fcntl
    verify_source();verify_inputs()
    lock=(OUT/'.submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=OUT/'submissions.json'
    state=read(ledger) if ledger.exists() else dict(datasets={},variants={},preserved_existing=read(OUT/'matrix.json')['preserved_existing'])
    def save():
        temp=OUT/'submissions.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(ledger)
    def sbatch(script,args,options):
        env=dict(os.environ,MIP_MATRIX_SOURCE=str(SOURCE))
        output=subprocess.check_output(['sbatch','--parsable','--chdir='+str(OUT)]+options+[str(SOURCE/script)]+args,text=True,env=env).strip()
        job=output.split(';')[0];assert job.isdigit();return job
    for i,(case,angle) in enumerate(DATASETS):
        k=f'{case}_wrr{angle}'
        if k not in state['datasets']:
            job=sbatch('job.sh',['export',str(i)],[f'--job-name=mip_export_{case}_w{angle}',f'--output={dataset_root(case,angle)}/logs/%x-%j.out',f'--error={dataset_root(case,angle)}/logs/%x-%j.err'])
            state['datasets'][k]=dict(job=job);save()
    m=read(COMPARISON/'inputs/manifest.json')
    for v in matrix():
        name=v['variant'];root=variant_root(v['case'],v['angle'],v['pc_convention'])
        entry=state['variants'].setdefault(name,{})
        task=m['cases'][v['case']]['production']['task'];tag='phys' if v['pc_convention']==CONVENTIONS[0] else 'bt'
        short=f'{v["case"].split("_case")[0]}_w{v["angle"]}_{tag}'
        opts=[f'--array={task}',f'--output={root}/logs/%x-%A_%a.out',f'--error={root}/logs/%x-%A_%a.err']
        if 'preflight_job' not in entry:
            dep=state['datasets'][f'{v["case"]}_wrr{v["angle"]}']['job']
            entry['preflight_job']=sbatch('preflight.sh',[str(root)],opts+[f'--job-name=mip_pf_{short}',f'--dependency=afterok:{dep}']);save()
        if 'full_job' not in entry:
            pf=entry['preflight_job']
            entry['full_job']=sbatch('full.sh',[str(root),pf],opts+[f'--job-name=mip_full_{short}',f'--dependency=afterok:{pf}']);save()
    print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['seal','stage','export','qualify','submit']);parser.add_argument('value',nargs='?')
    a=parser.parse_args()
    if a.mode=='seal':write(SOURCE/'source_lock.json',dict(files=files(SOURCE)))
    elif a.mode=='stage':stage()
    elif a.mode=='export':export_dataset(int(a.value))
    elif a.mode=='qualify':qualify(a.value)
    elif a.mode=='submit':submit()
