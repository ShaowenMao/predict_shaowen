"""Qualify only the campaign's managed path; reuse the completed numerical baseline."""
from pathlib import Path
import json,os,subprocess,sys,shutil,math,time,traceback
HERE=Path(__file__).resolve().parent
S=json.loads((HERE/'spec.json').read_text());BASE=Path(S['base'])
sys.path.insert(0,str(BASE));import context as C


def main():
    for n,h in C.read(HERE/'source_lock.json')['files'].items():assert C.sha(HERE/n)==h,n
    for n,h in C.read(BASE/'source_lock.json')['files'].items():assert C.sha(BASE/n)==h,n
    for p,h in S['inputs'].items():assert C.sha(p)==h,p
    C.verify_external(C.read(BASE/'spec.json'))
    old=Path(S['reference']);C.completion(old/'baseline')
    old_memory=C.read(old/'baseline_memory.json')
    assert old_memory['events']==0,'Reference must complete without memory-limit events'
    job=os.environ['SLURM_JOB_ID'];actual=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=45)
    a=dict(x.split('=',1) for x in actual.split() if '=' in x)
    for k,v in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',NumCPUs='2',MinMemoryNode='22G',Requeue='0').items():assert a[k]==v
    out=Path(S['root'])/'results'/('job'+job);out.mkdir(parents=True,exist_ok=False)
    C.write(out/'scheduler.json',dict(job=job,actual=actual))
    try:
        shutil.copytree(old/'baseline',out/'baseline')
        shutil.copytree(old/'planned60',out/'planned60')
        c=C.toml(old/'baseline.toml')
        item={k:c[k] for k in ('slot','raw','geometry','raw_hash','geometry_hash')}
        item.update(raw=str(out/'planned60/raw.bin'),geometry=str(out/'planned60/geometry.bin'))
        assert C.sha(item['raw'])==item['raw_hash'] and C.sha(item['geometry'])==item['geometry_hash'] and item['slot']==518
        C.config(out/'managed.toml',dict(**{k:c[k] for k in ('pipeline','pipeline_hash','bridge')},threads=2,passes=1,output=str(out/'managed')))
        with (out/'managed.toml').open('a') as f:
            f.write('\n[[cases]]\n')
            for k,v in item.items():f.write(k+' = '+(json.dumps(v) if isinstance(v,str) else str(v))+'\n')
        C.write(out/'reference.json',dict(source=str(old),baseline_completion_sha256=C.sha(old/'baseline/completion.toml'),
            numerical_baseline_reused=True,baseline_rerun=False,reference_memory=old_memory,
            original_24g_headroom_passed=old_memory['passed'],candidate_resource_scope='this managed-only job'))
        C.run(['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',str(BASE/'worker.jl'),str(out/'managed.toml')],out/'managed.log')
        numerical=C.compare(out);C.write(out/'numerical_review.json',numerical)
        exported=C.export(out);C.write(out/'export_review.json',exported)
        memory=C.memory_snapshot();C.write(out/'memory_review.json',memory)
        assert memory['limit_gib']==22 and memory['passed'],memory
        for p,h in S['inputs'].items():assert C.sha(p)==h,p
        C.verify_external(C.read(BASE/'spec.json'))
        ctx=dict(status='PASS_BROADER_NUMERICAL_CURVE',record=C.read(BASE/'spec.json')['records'][4],numerical=numerical,export=exported,
            artifacts={str(p.relative_to(out)):C.sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
            native_import_passed=False,campaign_launch_ready=False,production_modified=False)
        C.write(out/'context.json',ctx)
        C.write(out/'COMPLETE.json',dict(status='PASS_TARGETED_CORE518',core=518,job=job,source_lock=C.sha(HERE/'source_lock.json'),
            memory=memory,minimum_measured_headroom_gib=math.ceil(memory['required_gib']),
            memory_scope='fresh managed-only candidate job; baseline reused from separate completed job',
            context_sha256=C.sha(out/'context.json'),original_attempt=str(old),other_cores_changed=False,
            original_24g_headroom_reclassified=False,campaign_launch_ready=False))
        print(json.dumps(C.read(out/'COMPLETE.json')))
    except BaseException:
        C.write(out/'FAILURE.json',dict(job=job,error=traceback.format_exc(),memory=C.memory_snapshot()));raise


if __name__=='__main__':main()
