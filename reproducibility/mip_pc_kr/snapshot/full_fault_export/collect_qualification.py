"""Join completed receipts; do not rerun science or change any accepted table."""
import csv
import json
from pathlib import Path
import subprocess
import tomllib

from common import HERE,read,write,sha,verify,receipt,pins_ok


def launcher_review():
    path=HERE/'QUALIFIED_LAUNCHER_LOCK.json';lock=read(path)
    assert lock['base_lock_sha256']==sha(HERE/'LOCK.json')
    pins_ok({HERE/n:h for n,h in lock['files'].items()})
    evidence=[path]
    jobs=set()
    for name,version in [('probe-upscaling','julia version 1.12.6'),
                         ('probe-reservoir','julia version 1.10.4')]:
        p=HERE/'launcher_probes'/name/'PASS.json';x=read(p)
        assert x['passed'] and x['version']==version
        assert x['launcher_lock_sha256']==sha(path) and x['memory']['events']==0
        evidence.append(p);jobs.add(str(x['job']))
    p=HERE/'warm_load_probe/PASS.json';x=read(p)
    assert x['passed'] and x['zero_flow_solves'] and x['zero_mip_sweeps']
    assert x['load_path']=='@:@stdlib' and x['launcher_lock_sha256']==sha(path)
    assert x['memory']['events']==0
    evidence.append(p);jobs.add(str(x['job']))
    return evidence,jobs


def accounting_review(ids):
    fields=['JobID','State','ExitCode','Account','QOS','AllocCPUS','ReqMem','Elapsed','NodeList']
    raw=subprocess.check_output(['sacct','-j',','.join(sorted(ids)),'--allocations','-P',
                                 '--noheader','--format='+','.join(fields)],text=True,timeout=60)
    rows=[dict(zip(fields,line.split('|'),strict=True)) for line in raw.splitlines() if line]
    assert {r['JobID'] for r in rows}==ids
    for row in rows:
        assert row['State']=='COMPLETED' and row['ExitCode']=='0:0',row
        assert row['Account']==row['QOS']=='mit_amf_advanced_cpu',row
    return rows


def main():
    cfg=verify();receipt('final')
    final=read(HERE/'QUALIFICATION.json');assert final['passed']
    selector=HERE/'selector_crosscheck_v1';s=read(selector/'RESULT.json');assert s['passed']
    pins_ok(s['source_pins']);assert sha(selector/'PASS.toml')==s['pass_sha256']
    choices={r['slot']:r['indices'] for r in tomllib.loads((selector/'PASS.toml').read_text())['records']}
    p=read(HERE/'audit/PLAN.json')
    records=[]
    for row in p['placements']:
        owner=read(HERE/'owners'/row['owner']/'PASS.json');pins_ok(owner['artifacts'])
        assert owner['sample_ids']==choices[row['slot']]
        records.append(dict(slot=row['slot'],window=row['window'],slice=row['slice'],sample=row['sample'],
                            seed=row['seed'],owner=row['owner'],selected_states=len(owner['sample_ids']),
                            max_abs_krg_error=owner['max_abs_krg_error'],max_abs_krw_error=owner['max_abs_krw_error'],
                            approximation_review_flag=owner['approximation_review_flag']))
    assert len(records)==522
    auxiliary_native=tomllib.loads((HERE/'native/constitutive_pass.toml').read_text())
    native=tomllib.loads((HERE/'native_production/constitutive_pass.toml').read_text())
    full=tomllib.loads((HERE/'fullgrid/PASS.toml').read_text())
    version=read(HERE/'repairs/runtime_version_v1/VERSION_PASS.json')
    assert auxiliary_native['passed'] and native['passed'] and full['passed']
    assert auxiliary_native['julia']=='1.12.6' and native['julia']=='1.10.4'
    assert len(auxiliary_native['records'])==len(native['records'])==522
    assert version['passed'] and version['version']=='julia version 1.10.4'
    assert version['native_receipt_sha256']==sha(HERE/'native_production/constitutive_pass.toml')
    assert full['all_fault_tables_exact'] and full['host_hysteresis_unchanged']
    assert full['fault_hysteresis_disabled'] and not full['full_reservoir_simulation_run']
    launcher_evidence,jobs=launcher_review()
    for stage in [*final['stage_receipts'],'final']:
        x=receipt(stage);jobs.add(str(x['job']))
        assert x['memory']['events']==0
    # Include the finite controller and independent selector test, not failed attempts.
    jobs.update(['22865836','22865871'])
    accounting=accounting_review(jobs)
    before={p.name:sha(p) for p in (HERE/'intents').glob('*.json')}
    replay=subprocess.run(['python3.12',str(HERE/'qualified_dispatch.py')],capture_output=True,
                          text=True,check=True,timeout=120)
    after={p.name:sha(p) for p in (HERE/'intents').glob('*.json')}
    assert before==after and 'SUBMITTED ' not in replay.stdout
    peak={k:read(HERE/'receipts'/(k+'.json'))['memory']['peak_bytes']/1048576 for k in final['stage_receipts']}
    commit=subprocess.check_output(['git','-C',cfg['runtime'],'rev-parse','HEAD'],text=True).strip()
    clean=subprocess.check_output(['git','-C',cfg['runtime'],'status','--porcelain','--untracked-files=no'],text=True)
    assert commit==read(HERE/'snapshot.json')['runtime_commit'] and not clean
    with (HERE/'PLACEMENT_REVIEW.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    report=dict(passed=True,case=p['case'],placements=522,unique_inputs=p['scientific_unique'],
                derived_owners=len(p['owners']),all_julia_selection_indices_match=True,
                native_evaluation_points=sum(r['evaluation_points'] for r in native['records']),
                native_max_kr_error=max(max(r['max_errors'][:2]) for r in native['records']),
                native_max_pc_error_pa=max(r['max_errors'][2] for r in native['records']),
                native_versions_passed=['1.10.4','1.12.6'],
                native_endpoint_shift_exceptions=len(native['exact_documented_native_exceptions']),
                native_max_endpoint_krg=max(r['native_krg_zero'] for r in native['exact_documented_native_exceptions']),
                endpoint_exception_note='Pinned importer shifts the first knot by 1e-16; raw zero nodes are unchanged. No new exception or relaxed tolerance was introduced.',
                peak_memory_mib=peak,runtime_commit=commit,runtime_tracked_tree_clean=True,
                sampling_max_abs_krg_error=max(r['max_abs_krg_error'] for r in records),
                sampling_max_abs_krw_error=max(r['max_abs_krw_error'] for r in records),
                flagged_placements=sum(r['approximation_review_flag'] for r in records),
                all_pc100_preserved=True,selected_kr_exact=True,host_hysteresis_retained=True,
                fault_hysteresis_disabled=True,full_grid_preflight=full,
                completed_jobs=accounting,qualified_launcher_probes_passed=True,
                finished_dispatch_reentry_created_no_jobs=True,
                finished_dispatch_reentry_stdout=replay.stdout,
                portability_note='Production Julia 1.10.4 emitted HostCPUFeatures cached-feature warnings. Native and full-grid checks passed on the tested node; general cross-node reservoir-cache portability is not certified.',
                full_campaign_launched=False,reservoir_time_marching=False,production_modified=False,
                dispatcher_scope='Live bounded read-through, warm-worker, partial-batch restart, refill, resource admission, assembly dependencies',
                launch_gate_remaining='2430-case owner manifest/paused-ledger reconciliation and class allowlist in new namespace; no automatic launch',
                evidence={str(x):sha(x) for x in [HERE/'LOCK.json',HERE/'QUALIFICATION.json',HERE/'PLACEMENT_REVIEW.csv',
                         selector/'RESULT.json',HERE/'native/constitutive_pass.toml',HERE/'native_production/constitutive_pass.toml',
                         HERE/'repairs/runtime_version_v1/VERSION_PASS.json',HERE/'fullgrid/PASS.toml',
                         *launcher_evidence,Path(__file__)]})
    write(HERE/'COMPLETE_REVIEW.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evidence','full_grid_preflight','completed_jobs')},indent=2))


if __name__=='__main__':main()
