"""Prepare and submit isolated Pc-only MIP comparisons on Engaging.

Compatible with login-node Python 3.6. No numerical computation here.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'corrected_hysteresis_15case_20260911_v1'
PREFIX='gom_step62_effective_pc_global_plateau'

def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def readenv(path):
    return {s[7:].split('=',1)[0]:shlex.split(s.split('=',1)[1])[0]
            for s in path.read_text().splitlines() if s.startswith('export ')}

def command(args):return subprocess.check_output(args,universal_newlines=True).strip()

def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old),count)
    return text.replace(old,new)

def seal(dest):
    files=[p for p in dest.iterdir() if p.is_file() and p.name!='STUDY_SHA256.txt']
    files+=list((dest/'source').iterdir())
    files += [OLD/'hysteresis_validation.jl',OLD/'reference/advanced.jl',OLD/'reference/production_qoi.jl',ROOT/'launch.sh']
    (dest/'STUDY_SHA256.txt').write_text(''.join('{}  {}\n'.format(sha(p),p) for p in sorted(files) if p.is_file()))

def prepare():
    assert ROOT.name=='mip_pc_breakthrough_6case_20260916_v1'
    assert not (ROOT/'submissions.json').exists(),'Submitted study must remain immutable'
    records=json.loads((ROOT/'sources.json').read_text(encoding='utf-8-sig'))
    inputs={x['name']:x for x in json.loads((ROOT/'input_manifest.json').read_text())}
    runtime=OLD/'runtime'
    commit=command(['git','-C',str(runtime),'rev-parse','HEAD'])
    assert commit=='c403d9dd1bbb424ddfae9ae659d900c26c84e59d'
    assert not command(['git','-C',str(runtime),'status','--porcelain','--untracked-files=no'])
    matrix=[]
    for r in records:
        ref=r['reference'];old=Path(ref['root']);name=ref['name']+'_pczero'
        dest=ROOT/'variants'/name
        assert not dest.exists(),'Refuse to overwrite prepared variant: '+str(dest)
        candidate=ROOT/'inputs'/ref['name']/'candidate_specific.mat'
        assert sha(candidate)==inputs[ref['name']]['sha256']
        audit=candidate.parent/'pc_only_audit.json'
        checked=json.loads(audit.read_text());assert checked['passed'] and checked['output_sha256']==sha(candidate)
        for pin in r['inputs'].values():assert sha(pin['path'])==pin['sha256']
        oldenv=readenv(old/'locked.env');env=dict(oldenv)
        assert oldenv['GOM_PRODUCTION_JUTULDARCY_COMMIT']==commit
        dest.mkdir(parents=True)
        shutil.copytree(str(old/'source'),str(dest/'source'))
        source_checks=[]
        for path in (dest/'source').iterdir():
            if path.suffix not in ('.jl','.sh','.sbatch'):continue
            before=path.read_text();normalized=before.replace(str(old),str(dest));text=normalized
            if path.name=='pair_input_audit.jl':text=(ROOT/'pair_input_audit.jl').read_text()
            if path.name=='gom_step62_production_final_check.jl':
                text=replace(text,'expected_restarts == CURRENT_RESTARTS\n)',
                    'expected_restarts == CURRENT_RESTARTS ||\n    expected_restarts == collect(1:EXPECTED_STEPS)\n)')
                text=replace(text,'$LEGACY_RESTARTS (legacy recovery) or $CURRENT_RESTARTS (current).',
                    '$LEGACY_RESTARTS (legacy recovery), $CURRENT_RESTARTS (current), or all 1:210 report states.')
            path.write_text(text)
            assert text==normalized or path.name in ('pair_input_audit.jl','gom_step62_production_final_check.jl')
            source_checks.append(dict(file=path.name,old_sha256=sha(old/'source'/path.name),new_sha256=sha(path),
                                      same_after_launcher_path_relocation=text==normalized))
            if path.suffix in ('.sh','.sbatch'):subprocess.check_call(['bash','-n',str(path)])
        campaign_id=name+'_20260916'
        manifest=(old/'campaign.toml').read_text()
        manifest=manifest.replace(str(old),str(dest))
        manifest=replace(manifest,json.dumps(oldenv['GOM_PRODUCTION_CAMPAIGN_ID']),json.dumps(campaign_id))
        manifest=replace(manifest,json.dumps(oldenv['GOM_PRODUCTION_SPECIFIC_MAT']),json.dumps(str(candidate)))
        manifest=replace(manifest,json.dumps(oldenv['GOM_PRODUCTION_SPECIFIC_SHA256']),json.dumps(sha(candidate)))
        manifest=replace(manifest,'specific_bytes = '+oldenv['GOM_PRODUCTION_SPECIFIC_BYTES'],
                         'specific_bytes = '+str(candidate.stat().st_size))
        manifest=replace(manifest,'pc_convention = "physical_prebreakthrough"',
                         'pc_convention = "zero_before_breakthrough_pc_only"')
        experiment={'reference_campaign':str(old/'campaign.toml'),'reference_campaign_sha256':sha(old/'campaign.toml'),
            'reference_full_job':str(ref['full_job']),'reference_specific':r['inputs']['mip']['path'],
            'reference_specific_sha256':r['inputs']['mip']['sha256'],
            'zero_pc_source':r['inputs']['zero']['path'],'zero_pc_sha256':r['inputs']['zero']['sha256'],
            'physical_pc_source':r['inputs']['physical']['path'],'physical_pc_sha256':r['inputs']['physical']['sha256'],
            'pc_only_audit_sha256':sha(audit),'kr_policy':'finished_mip_values_unchanged',
            'solver_policy':'same_corrected_runtime_solver_tolerances_and_timestep_controls',
            'start_policy':'time_zero_no_restart_reuse',
            'retention_checker_change':'accept exactly all 210 states; existing state and QoI checks retained'}
        manifest+='\n[pc_only_experiment]\n'+''.join(k+' = '+json.dumps(v)+'\n' for k,v in experiment.items())
        (dest/'campaign.toml').write_text(manifest)
        manifest_sha=sha(dest/'campaign.toml')
        (dest/'campaign.toml.sha256').write_text(manifest_sha+'  campaign.toml\n')
        env.update(GOM_PRODUCTION_CAMPAIGN_ID=campaign_id,
            GOM_PRODUCTION_MANIFEST_PATH=str(dest/'campaign.toml'),
            GOM_PRODUCTION_MANIFEST_COMPANION_PATH=str(dest/'campaign.toml.sha256'),
            GOM_PRODUCTION_MANIFEST_SHA256=manifest_sha,GOM_PRODUCTION_ARCHIVE_ROOT=str(dest),
            GOM_PRODUCTION_SPECIFIC_MAT=str(candidate),GOM_PRODUCTION_SPECIFIC_SHA256=sha(candidate),
            GOM_PRODUCTION_SPECIFIC_BYTES=str(candidate.stat().st_size),
            MIP_REFERENCE_SPECIFIC_MAT=r['inputs']['mip']['path'],
            PHYSICAL_PC_SPECIFIC_MAT=r['inputs']['physical']['path'],ZERO_PC_SPECIFIC_MAT=r['inputs']['zero']['path'])
        (dest/'locked.env').write_text(''.join('export {}={}\n'.format(k,shlex.quote(v)) for k,v in env.items()))
        envdiff={k:{'old':oldenv.get(k),'new':env.get(k)} for k in set(oldenv)|set(env) if oldenv.get(k)!=env.get(k)}
        allowed={'GOM_PRODUCTION_CAMPAIGN_ID','GOM_PRODUCTION_MANIFEST_PATH','GOM_PRODUCTION_MANIFEST_COMPANION_PATH',
            'GOM_PRODUCTION_MANIFEST_SHA256','GOM_PRODUCTION_ARCHIVE_ROOT','GOM_PRODUCTION_SPECIFIC_MAT',
            'GOM_PRODUCTION_SPECIFIC_SHA256','GOM_PRODUCTION_SPECIFIC_BYTES','MIP_REFERENCE_SPECIFIC_MAT',
            'PHYSICAL_PC_SPECIFIC_MAT','ZERO_PC_SPECIFIC_MAT'}
        assert set(envdiff)<=allowed
        (dest/'environment_comparison.json').write_text(json.dumps(envdiff,indent=2)+'\n')
        (dest/'source_comparison.json').write_text(json.dumps(source_checks,indent=2)+'\n')
        pins={}
        for line in (old/'INPUTS_SHA256.txt').read_text().splitlines():
            digest,path=line.split(None,1);path=path.lstrip('*')
            if not Path(path).is_absolute():path=str((old/path).resolve())
            pins[path]=digest
        for pin in r['inputs'].values():pins[pin['path']]=pin['sha256']
        pins[str(candidate)]=sha(candidate);pins[str(audit)]=sha(audit)
        (dest/'INPUTS_SHA256.txt').write_text(''.join('{}  {}\n'.format(d,p) for p,d in sorted(pins.items())))
        (dest/'provenance.json').write_text(json.dumps(experiment,indent=2)+'\n')
        seal(dest)
        matrix.append(dict(name=name,risk=ref['risk'],angle=ref['angle'],case=ref['case'],task=ref['task'],
            root=str(dest),reference_root=str(old),reference_full_job=ref['full_job'],specific=str(candidate),
            specific_sha256=sha(candidate),runtime_commit=commit,manifest_sha256=manifest_sha))
        print('PREPARED',name,flush=True)
    (ROOT/'matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
    (ROOT/'logs').mkdir(exist_ok=True)
    with tarfile.open(str(ROOT/'prepared_launchers.tgz'),'w:gz') as archive:
        archive.add(str(ROOT/'matrix.json'),arcname='matrix.json')
        archive.add(str(ROOT/'variants'),arcname='variants')
    with tarfile.open(str(ROOT/'reference_configs.tgz'),'w:gz') as archive:
        for r in records:
            ref=r['reference']
            for filename in ('campaign.toml','locked.env'):
                archive.add(str(Path(ref['root'])/filename),arcname='variants/'+ref['name']+'/'+filename)

def submit():
    assert (ROOT/'local_qualification.json').is_file(),'Local manifest/source qualification required'
    q=json.loads((ROOT/'local_qualification.json').read_text());assert q['passed']
    assert q['matrix_sha256']==sha(ROOT/'matrix.json')
    ledger=ROOT/'submissions.json'
    assert not ledger.exists(),'Never blindly resubmit an existing ledger'
    rows=json.loads((ROOT/'matrix.json').read_text());submitted=[]
    ledger.write_text('[]\n')
    for r in rows:
        def launch(kind,dependency=None):
            cmd=['sbatch','--parsable','--partition=mit_normal','--account=mit_amf_advanced_cpu',
                '--qos=mit_amf_advanced_cpu','--cpus-per-task=8','--mem='+('12G' if kind=='preflight' else '18G'),
                '--time='+('04:00:00' if kind=='preflight' else '4-00:00:00'),'--no-requeue',
                '--array='+str(r['task']),'--job-name=mpz_'+r['risk'][:3]+'_'+str(r['angle'])+'_'+kind[:2],
                '--output='+str(ROOT/'logs'/(r['name']+'_'+kind+'-%A_%a.out')),
                '--error='+str(ROOT/'logs'/(r['name']+'_'+kind+'-%A_%a.err'))]
            if dependency:cmd+=['--dependency=afterok:'+dependency]
            cmd+=[str(ROOT/'launch.sh'),r['root'],kind]
            if dependency:cmd.append(dependency)
            job=command(cmd).split(';')[0];assert job.isdigit(),job
            return job
        x=dict(r);x['preflight_job']=launch('preflight')
        submitted.append(x);ledger.write_text(json.dumps(submitted,indent=2)+'\n')
        x['full_job']=launch('full',x['preflight_job'])
        ledger.write_text(json.dumps(submitted,indent=2)+'\n')
        print(x['name'],x['preflight_job'],x['full_job'],flush=True)
    receipts=[]
    for x in submitted:
        for key in ('preflight_job','full_job'):
            rec=command(['scontrol','show','job','-o',x[key]])
            assert 'Account=mit_amf_advanced_cpu' in rec and 'QOS=mit_amf_advanced_cpu' in rec
            assert 'NumCPUs=8' in rec and ('mem=12G' if key=='preflight_job' else 'mem=18G') in rec
            receipts.append(rec)
    (ROOT/'scheduler_verification.txt').write_text('\n'.join(receipts)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','submit'])
    args=ap.parse_args()
    if args.action=='prepare':prepare()
    else:submit()
