"""Complete one fault, preserving old science and every accepted new owner."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib
import traceback

from common import HERE, read, write, sha, pins_ok, verify, receipt
from planning import digest
from resource import cgroup, snapshot, passed, parse_allocation
from reuse import fingerprint
import reuse
from bounded_io import normalized_file, scientific_hash, discard_consumed_pages

reuse.normalized_file = normalized_file


def canonical_equal(a, b, geometry=False, raw_normal=None):
    n = 104 if geometry else 96
    with Path(a).open('rb') as x, Path(b).open('rb') as y:
        h1,h2 = bytearray(x.read(n)),bytearray(y.read(n))
        h1[8:16]=h2[8:16]=bytes(8)
        if geometry:
            h1[72:104]=h2[72:104]=bytes.fromhex(raw_normal)
        assert h1==h2
        while True:
            aa,bb=x.read(1024**2),y.read(1024**2)
            assert aa==bb, 'Hash match but different canonical bytes'
            discard_consumed_pages(x);discard_consumed_pages(y)
            if not aa:break


def audit(cfg):
    old=Path(cfg['old_root']);tag=cfg['old_release']
    final=old/'final'/tag/'QUALIFICATION.json'
    assert sha(final)==cfg['old_final_sha256']
    f=read(final);assert f['status']=='PASS_REFERENCE_END_TO_END' and f['all_522_placements']
    for item in f['receipts']:
        assert sha(item['path'])==item['sha256']
    a=read(old/'audits'/tag/'FULL_AUDIT.json')
    raw_manifest=read(old/'exports'/tag/'raw_manifest.json')
    assert len(a['records'])==len(raw_manifest['records'])==522
    assert not a['incomplete'] and not a['resource_review']
    scenario=raw_manifest['scenario']
    for k,v in dict(clay_theta_deg=60,clay_swi=.2,clay_endpoint=.8,clay_nw=3.5,clay_ng=2.5,
                    sand_endpoint=.95,wetting_rule='krw_strictly_positive',cell_based_pc=True,flow_axis=3).items():
        assert scenario[k]==v,(k,scenario[k])
    pilot=read(Path(cfg['pilot'])/'manifest.json')
    pins_ok({str(Path(cfg['pilot'])/k):v for k,v in pilot['sources'].items()})
    pins_ok(pilot['pins'])
    snap=read(HERE/'snapshot.json');pins_ok(snap['pins'])
    for row in snap['files'].values():pins_ok({row['path']:row['sha256']})
    method=digest(dict(pipeline=pilot['pipeline_hash'],bridge=sha(pilot['bridge']),axis=3,threads=2,
                       scenario=scenario,selector=sha(HERE/'planning.py'),kernel=sha(Path(cfg['pilot'])/'Kernel.jl')))
    owners={};placements=[];scientific={};source_pins={}
    for r in sorted(raw_manifest['records'],key=lambda r:r['slot']):
        slot=r['slot'];v=a['records'][str(slot)];e=v['entry'];s=v['summary']
        assert slot==(r['window']-1)*87+r['slice'] and 1<=slot<=522
        assert v['historical']['passed'] and v['historical']['bitwise_file_equal']
        assert s['points']==100 and s['axis']==3 and s['julia_threads']==s['blas_threads']==2
        assert s['source_tree_sha256']==pilot['pipeline_hash'] and s['bridge_sha256']==sha(pilot['bridge'])
        assert sha(e['accepted_path'])==e['accepted_sha256']
        accepted=read(e['accepted_path']);assert accepted['numerical_checks_passed']
        folder=Path(e['attempt'])/'numerical/planned'
        raw=folder/'raw.bin';geo=folder/'geometry.bin';scenario_path=folder/'scenario_audit.toml'
        expected=e['expected_input_hashes']
        fp=fingerprint(raw,geo,slot,expected['raw.bin'],expected['geometry.bin'],method)
        sa=tomllib.loads(scenario_path.read_text())
        assert sa['passed'] and sa['core']==slot and sa['clay_swi']==.2 and sa['clay_ng']==2.5 and sa['clay_nw']==3.5
        assert sa['contact_angle_ratio']==0.5773502691896258 and sa['clay_reference_anchor_pa']==s['model_audit']['clay_reference_anchor_pa']
        assert sa['entry_pressure_pa']==r['entry_pressure_pa']
        assert sa['unchanged_geometry_intrinsic_properties_sand_and_sampling'] and not sa['new_curve_fit_performed']
        rel='numerical/planned/scenario_audit.toml'
        assert accepted['outputs'][rel]==sha(scenario_path)
        curve=Path(v['reference'])/'pc_kr_raw.csv';flow=Path(v['reference'])/'flow_calls.jsonl'
        pins_ok({curve:v['curve_hash']})
        assert scientific_hash(flow)==v['flow_hash']
        completed=tomllib.loads((flow.parent/'completion.toml').read_text())
        assert completed['execution_complete'] and completed['points']==100
        pins_ok({flow.parent/n:h for n,h in completed['outputs'].items()})
        assert s['raw_input_sha256']==expected['raw.bin'] and s['geometry_sha256']==expected['geometry.bin']
        if fp['numerical_key'] in scientific:
            prev=scientific[fp['numerical_key']]
            canonical_equal(raw,prev['raw'])
            canonical_equal(geo,prev['geometry'],True,fp['raw_scientific_sha256'])
        else:
            scientific[fp['numerical_key']]=dict(raw=str(raw),geometry=str(geo))
        # Keep different archived floating-point representations separate even if physics is identical.
        key=digest(dict(scientific=fp['numerical_key'],curve=v['curve_hash'],entry=r['entry_pressure_pa']))
        if key not in owners:
            owners[key]=dict(key=key,slot=slot,curve=str(curve),curve_sha256=v['curve_hash'],
                             entry_pressure_pa=r['entry_pressure_pa'],fingerprint=fp,raw=str(raw),geometry=str(geo))
        placements.append(dict(r,owner=key,fingerprint=fp))
        source_pins[str(curve)]=v['curve_hash']
        if slot%20==0:print('AUDIT_PROGRESS',slot,522,flush=True)
    assert [r['slot'] for r in placements]==list(range(1,523))
    keys=sorted(owners)
    plan=dict(passed=True,case='s03_c022_case12',placements=placements,owners=owners,
              scientific_unique=len(scientific),existing_curve_unique=len(owners),
              batches=[keys[i:i+32] for i in range(0,len(keys),32)],source_pins=source_pins,
              old_manifest=raw_manifest,method_sha256=method,full100_recomputed=False,
              full_campaign_enabled=False,production_modified=False)
    write(HERE/'audit/PLAN.json',plan)
    return [HERE/'audit/PLAN.json']


def logged(cmd,path,env=None,timeout=3600):
    print('PROCESS_BEGIN',cmd[0],path,flush=True)
    with Path(path).open('x') as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True,timeout=timeout)
    print('PROCESS_PASS',path,flush=True)


def matlab(cfg,name,args,out):
    expr="addpath('"+str(HERE)+"');"+name+'('+','.join("'"+str(x).replace("'","''")+"'" for x in args)+');'
    logged([cfg['matlab'],'-batch',expr],out,timeout=3300)


def assemble(cfg):
    p=read(HERE/'audit/PLAN.json');pins_ok(p['source_pins'])
    out=HERE/'assembly';out.mkdir()
    records=[]
    for row in p['placements']:
        k=row['owner'];folder=HERE/'owners'/k;mark=folder/'PASS.json';r=read(mark)
        assert r['passed'] and r['key']==k and r['original_pc100_exact'] and r['pc_export_exact']
        pins_ok(r['artifacts'])
        records.append({**{n:row[n] for n in ('slot','window','slice','sample','seed','entry_pressure_pa')},
                        'curve':str(folder/'paired.csv'),'audit':str(mark)})
    scenario=dict(p['old_manifest']['scenario'],kr_flow_states=10,pc_saved_states=100,
                  kr_interpolation='piecewise_linear_actual_selected_sg',original_pc100_retained=True,
                  physical_prebreakthrough=True,derivation='accepted_raw_100_state_results')
    manifest=dict(p['old_manifest'],records=records,scenario=scenario,source=sha(HERE/'LOCK.json'))
    write(out/'raw_manifest.json',manifest);write(out/'scenario.json',scenario)
    with (out/'entries.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['slot','entry_pressure_pa'])
        for r in records:w.writerow([r['slot'],format(r['entry_pressure_pa'],'.17g')])
    matlab(cfg,'verify_matlab_startup',[out/'startup.json'],out/'startup.log')
    assert read(out/'startup.json')['passed']
    matlab(cfg,'assemble60',[out/'raw_manifest.json',out/'raw'],out/'raw_assembly.log')
    raw=out/'raw/candidate_raw_s03_c022_case12_wrr60.mat'
    logged([cfg['export_python'],'-B',str(HERE/'build_tables.py'),str(raw),str(out/'entries.csv'),str(out/'tables')],out/'tables.log')
    tables=out/'tables/candidate_tables_s03_c022_case12.mat'
    specific=out/'candidate_specific_s03_c022_case12_pc100_kr10.mat'
    matlab(cfg,'make_specific60',[cfg['baseline_specific'],tables,out/'tables/audit.json',out/'scenario.json',specific],out/'specific.log')
    assert read(out/'raw/roundtrip.json')['passed']
    assert read(str(specific)+'.native.mat.roundtrip.json')['passed']
    pins_ok(read(HERE/'snapshot.json')['pins'])
    report=dict(passed=True,placements=522,raw=str(raw),tables=str(tables),specific=str(specific),
                input_physics_unchanged=True,pc100_unchanged=True,kr10_only_change=True,
                sampling_errors='owners/*/PASS.json',full_reservoir_simulation_run=False,
                production_modified=False)
    write(out/'PASS.json',report)
    return [out/'PASS.json',raw,tables,specific,out/'tables/audit.json']


def runtime_env(cfg,stage):
    env=dict(os.environ,CORRECTED_STUDY_ROOT=str(HERE))
    depot=HERE/'depots'/stage;depot.mkdir(parents=True,exist_ok=True)
    env['JULIA_DEPOT_PATH']=str(depot)+':/home/shaowen/.julia'
    return env


def native(cfg):
    p=read(HERE/'audit/PLAN.json')
    tables=[str(HERE/'owners'/r['owner']/'sgof.csv') for r in p['placements']]
    assert len(tables)==522
    cmd=[cfg['julia'],'--startup-file=no','--threads=1','--gcthreads=1',
         '--compiled-modules=existing','--pkgimages=existing','--project='+cfg['runtime'],
         str(HERE/'native.jl'),str(HERE/'native'),*tables]
    logged(cmd,HERE/'logs/native_process.log',runtime_env(cfg,'native'),2500)
    mark=HERE/'native/constitutive_pass.toml'
    r=tomllib.loads(mark.read_text());assert r['passed'] and len(r['records'])==522
    return [mark]


def fullgrid(cfg):
    a=read(HERE/'assembly/PASS.json')
    cmd=[cfg['julia'],'--startup-file=no','--threads=2','--gcthreads=1',
         '--compiled-modules=existing','--pkgimages=existing','--project='+cfg['runtime'],
         str(HERE/'runtime_reference_preflight.jl'),cfg['common'],a['specific'],str(HERE/'fullgrid')]
    pins_ok(read(HERE/'snapshot.json')['pins'])
    logged(cmd,HERE/'logs/fullgrid_process.log',runtime_env(cfg,'fullgrid'),5200)
    mark=HERE/'fullgrid/PASS.toml';assert tomllib.loads(mark.read_text())['passed']
    pins_ok(read(HERE/'snapshot.json')['pins'])
    return list((HERE/'fullgrid').rglob('*.toml'))


def warm(cfg):
    pilot=Path(cfg['pilot']);m=read(pilot/'manifest.json');pins=read(pilot/'PREPARED.json')['pins'];pins_ok(pins)
    # Same source-pinned worker and prepared cache, invoked by the new dispatcher.
    conf=tomllib.loads(next((pilot/'configs').glob('small_*.toml')).read_text())
    out=HERE/'warm';conf['output']=str(out);conf['memory_mib']=2816
    conf['source_lock_sha256']=sha(HERE/'LOCK.json')
    for c in conf['cases']:
        c.pop('recover_artifacts',None);c.pop('reuse_receipt',None)
    text=''
    for k,v in conf.items():
        if k not in ('cases','pins'):text+=k+' = '+json.dumps(v)+'\n'
    for row in conf['cases']:
        text+='\n[[cases]]\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in row.items())
    text+='\n[pins]\n'+''.join(json.dumps(k)+' = '+json.dumps(v)+'\n' for k,v in pins.items())
    cfile=HERE/'warm_config.toml'
    with cfile.open('x') as f:f.write(text)
    env=dict(os.environ);dep=HERE/'depots/warm';dep.mkdir(parents=True,exist_ok=True)
    env['JULIA_DEPOT_PATH']=':'.join(map(str,[dep,pilot/'cache/depot',m['depot']]))
    cmd=[m['julia'],'--startup-file=no','--math-mode=ieee','--check-bounds=yes','--threads=2',
         '--gcthreads=1','--cpu-target=generic','--project='+m['project'],
         '--compiled-modules=strict','--pkgimages=existing',str(pilot/'worker.jl'),str(cfile)]
    logged(cmd,HERE/'logs/warm_process.log',env,1600)
    mark=out/'BATCH_PASS.toml';r=tomllib.loads(mark.read_text());assert r['passed'] and len(r['records'])==4
    paths=[mark]
    for item in r['records']:
        assert sha(item['receipt'])==item['receipt_sha256']
        d=tomllib.loads(Path(item['receipt']).read_text());assert d['passed']
        pins_ok({Path(item['receipt']).parent/k:v for k,v in d['artifacts'].items()})
        paths.append(Path(item['receipt']))
    pins_ok(pins)
    return paths


def final(cfg):
    from dispatch import dag
    stages={k:receipt(k) for k in dag() if k!='final'}
    p=read(HERE/'audit/PLAN.json');owners=[read(HERE/'owners'/k/'PASS.json') for k in p['owners']]
    assert len(p['placements'])==522 and all(r['selected_kr_exact'] and r['pc_export_exact'] for r in owners)
    prefix=read(HERE/'batches/prefix.json');resumed=read(HERE/'batches/derive000.json')
    assert all(r['reused'] for r in resumed['records'][:len(prefix['records'])])
    assert [r['receipt_sha256'] for r in prefix['records']]==[r['receipt_sha256'] for r in resumed['records'][:len(prefix['records'])]]
    report=dict(passed=True,status='PASS_ONE_FAULT_AND_BOUNDED_DISPATCHER',case=p['case'],placements=522,
                unique_scientific_inputs=p['scientific_unique'],derived_owners=len(owners),
                selected_states_range=[min(len(r['sample_ids']) for r in owners),max(len(r['sample_ids']) for r in owners)],
                max_abs_krw_sampling_error=max(r['max_abs_krw_error'] for r in owners),
                max_abs_krg_sampling_error=max(r['max_abs_krg_error'] for r in owners),
                sampling_flags=sum(r['approximation_review_flag'] for r in owners),
                pc100_preserved=True,complete_522_mapping_imported=True,
                native_values_and_derivatives_passed=True,full_grid_no_time_marching=True,
                warm_worker_dispatch_passed=True,partial_batch_recovery_passed=True,
                dispatcher_test_receipt=read(HERE/'tests.json'),
                stage_receipts={k:sha(HERE/'receipts'/(k+'.json')) for k in stages},
                memory={k:v['memory'] for k,v in stages.items()},
                campaign_launch_approved=False,production_modified=False,
                remaining='2430-case owner manifest and qualified resource-class admission; reservoir QoI impact not tested here')
    write(HERE/'QUALIFICATION.json',report)
    return [HERE/'QUALIFICATION.json']


def run(stage):
    from dispatch import dag
    cfg=verify();pol=dag()[stage]
    for dep in pol['deps']:receipt(dep)
    actual=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True,timeout=30)
    parse_allocation(actual,pol['cpus'],pol['memory_mib'])
    group=cgroup(os.environ['SLURM_JOB_ID']);start=time.monotonic()
    if stage=='prefix' or stage.startswith('derive'):
        logged([cfg['export_python'],'-B',str(HERE/'sparse.py'),stage],HERE/'logs'/(stage+'_process.log'),timeout=1000)
        paths=[HERE/'batches'/(stage+'.json')]
        for r in read(paths[0])['records']:
            assert sha(r['receipt'])==r['receipt_sha256'];paths.append(Path(r['receipt']))
    else:paths=globals()[stage](cfg)
    verify();mem=snapshot(group)
    stat=group/('memory.stat')
    details=dict(line.split() for line in stat.read_text().splitlines()) if stat.exists() else {}
    value=dict(passed=passed(mem,pol['memory_mib']),stage_work_complete=True,stage=stage,
               job=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start,memory=mem,
               lock_sha256=sha(HERE/'LOCK.json'),actual_scontrol=actual,
               artifacts={str(p):sha(p) for p in paths},production_modified=False,
               end_memory_stat=details,process_status=Path('/proc/self/status').read_text())
    write(HERE/'receipts'/(stage+'.json'),value)
    assert value['passed'],'Targeted stage memory review required; scientific outputs preserved'
    print('STAGE_PASS',stage,json.dumps(mem),flush=True)


if __name__=='__main__':
    stage=sys.argv[1]
    try:run(stage)
    except Exception:
        write(HERE/'failures'/(stage+'_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(stage=stage,error=traceback.format_exc()))
        raise
