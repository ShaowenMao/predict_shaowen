"""Build an isolated v4 from pinned v3 plus the qualified drainage cache."""
import ast
import json
from pathlib import Path
import shutil
import zipfile
import hashlib

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'phase1_pc40_adaptive10_runner_20261030_v3'
CACHE=ROOT.parent/'phase1_adaptive_statecache_20261030_v1'
OLD_PIN='4497359d39e01a47ce830ca4495a9682ba330b210eaa0856b785c45761d551c5'
CACHE_PIN='c1d92f53286c7444c3c0df09080654936405ee4eb4596c94f94dcab4832b1d0b'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,sort_keys=True,indent=2)
def replace(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)

def main():
    assert not (ROOT/'LOCK.json').exists()
    assert sha(OLD/'LOCK.json')==OLD_PIN and sha(CACHE/'LOCK.json')==CACHE_PIN
    old=read(OLD/'LOCK.json');cache=read(CACHE/'LOCK.json')
    shutil.copyfile(OLD/'LOCK.json',ROOT/'BASELINE_LOCK.json')
    excluded={'setup.py','seal.py','qualification_operator.py','STAGED.json'}
    for n,h in old['code'].items():
        assert sha(OLD/n)==h,n
        if n in excluded:continue
        p=ROOT/n;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            assert n in ('flow_packet.jl','worker.py','job.sh') or sha(p)==h,p
        else:shutil.copyfile(OLD/n,p)
    assert sha(OLD/'science_bundle.zip')==old['science_bundle_sha256']
    science=ROOT/'science';science.mkdir(exist_ok=True)
    with zipfile.ZipFile(OLD/'science_bundle.zip') as z:
        assert set(z.namelist())==set(old['science'])
        for n,h in old['science'].items():
            data=z.read(n);assert hashlib.sha256(data).hexdigest()==h
            p=science/n;p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert sha(p)==h
            else:
                with p.open('xb') as f:f.write(data)
    for n in ('DrainageCache.jl','CachedAdaptiveCurve.jl'):
        assert sha(CACHE/n)==cache['files'][n]
        shutil.copyfile(CACHE/n,science/n)
    for label in ('slow27','slow48','slow68'):
        passed=read(CACHE/'results'/label/'PASS.json')
        assert passed['passed'] and passed['candidate']==CACHE_PIN and passed['qualification_only']
    # No frozen source is edited. These transformations apply only to the new copy.
    p=ROOT/'flow_packet.jl';text=(OLD/'flow_packet.jl').read_text()
    text=replace(text,'include(joinpath(SRC,"AdaptiveCurve.jl"))',
                 'include(joinpath(SRC,"DrainageCache.jl"))\ninclude(joinpath(SRC,"CachedAdaptiveCurve.jl"))')
    text=replace(text,'    pc,kr=run_adaptive(x,g,output)',
                 '''    identity=(core=r["key"],raw_sha256=proof["raw_sha256"],
              geometry_sha256=proof["geometry_unplanned_sha256"],method=cfg["method"])
    pc,kr=run_cached_adaptive(x,g,output,identity)''')
    text=replace(text,'    receipt=Dict("passed"=>true,',
                 '''    @assert s["cached_states_verified"]==100 && s["cached_queries_verified"]==length(kr)
    @assert s["replay_states_checked"]==0 && s["reference_states_checked"]==0
    @assert s["state_access"]=="verified_per_core_cache_v1" && s["cache_payload_bytes"]<=512*1024^2
    implementation=Dict(n=>bytes2hex(open(sha256,joinpath(SRC,n))) for n in
        ("DrainageCache.jl","CachedAdaptiveCurve.jl"))
    receipt=Dict("state_cache_implementation"=>implementation,"passed"=>true,''')
    p.write_text(text,encoding='ascii',newline='\n')
    p=ROOT/'worker.py';text=(OLD/'worker.py').read_text()
    text=replace(text,'import planning\n','import planning\nfrom cache_contract import verify_cache_evidence\n')
    text=replace(text,"    return p\n\ndef stage_science", "    verify_cache_evidence(s,p,source,C.sha(C.CODE/'LOCK.json'),C.read(C.CODE/'LOCK.json')['science'])\n    return p\n\ndef stage_science")
    p.write_text(text,encoding='ascii',newline='\n')
    p=ROOT/'job.sh';text=(OLD/'job.sh').read_text()
    text=replace(text,'case "$mode" in','case "$mode" in\nintegration) exec /usr/bin/python3.12 -B "$CODE/packet_qualification.py" "$@" ;;')
    p.write_text(text,encoding='ascii',newline='\n')
    source=read(ROOT/'PILOT_INPUTS.json');case=source['case_id']
    oldplan=read(ROOT/'ORIGINAL_PLAN.json')
    assert sha(ROOT/'ORIGINAL_PLAN.json')==source['plan_sha256']
    records={r['slot']:r for r in oldplan['records']}
    selected=[records[i] for i in source['slots']]
    assert len(selected)==len({r['slot'] for r in selected})==32
    plan=dict(case_id=case,records=oldplan['records'],packets=[selected],allowed_packets=[0],purpose='qualification')
    put(ROOT/'plans'/f'{case}.json',plan)
    put(ROOT/'STAGED.json',dict(configuration_sha256=old['configuration_sha256'],method_sha256=old['method_sha256'],
         pilot_cases=[case],full_campaign_authorized=False,predecessor=OLD_PIN,state_cache=CACHE_PIN))
    # Keep the unchanged configuration available for offline identity regression.
    shutil.copytree(OLD/'configuration',ROOT/'configuration')
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    files={p.relative_to(science).as_posix():sha(p) for p in sorted(science.rglob('*')) if p.is_file()}
    with zipfile.ZipFile(ROOT/'science_bundle.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n in files:z.write(science/n,n)
    code={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.iterdir() if p.suffix in ('.py','.jl','.m','.sh')}
    for p in (ROOT/'matlab').glob('*.m'):code[p.relative_to(ROOT).as_posix()]=sha(p)
    for n in ('RUNTIME_BUNDLE.json','runtime_inventory.json','DATA_PINS.json','EXECUTABLE_PINS.json',
              'STAGED.json','replay_metadata.csv','PILOT_INPUTS.json','ORIGINAL_PLAN.json','BASELINE_LOCK.json'):
        code[n]=sha(ROOT/n)
    put(ROOT/'LOCK.json',dict(code=code,science=files,science_bundle_sha256=sha(ROOT/'science_bundle.zip'),
        configuration_sha256=old['configuration_sha256'],method_sha256=old['method_sha256'],
        plans={case:sha(ROOT/'plans'/f'{case}.json')},full_campaign_authorized=False,
        predecessor=OLD_PIN,state_cache_qualification=CACHE_PIN))
    print('SEALED',sha(ROOT/'LOCK.json'))

if __name__=='__main__':main()
