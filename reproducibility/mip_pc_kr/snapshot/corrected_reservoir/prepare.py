"""Build isolated matched launchers without editing any source campaign or input.

Use on Engaging with Python 3.12. All expensive model checks run in Slurm.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
ORIGINAL_RUNTIME = Path('/home/shaowen/orcd/scratch/gom_grid/code/immutable/JutulDarcy_ed81af8e')
RUNTIME = ROOT / 'runtime'
MATRIX = BASE / 'comparative_3case_wrr30_60_pc_conventions_20260910_v1/reservoir_matrix_launch_20260911_v1'
CASES = [('low','s01_c018_case07',262),('intermediate','s03_c022_case12',1137),('high','s03_c003_case08',848)]
BASE_COMMIT = 'ed81af8e88fcd2a7720f26872eef1d16d63469f5'
PREFIX = 'gom_step62_effective_pc_global_plateau'
TEMPLATE = MATRIX / 'variants/s01_c018_case07_wrr30_physical_prebreakthrough'

def sha(path):
    with open(path,'rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def readenv(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith('export '):
            key, value = line[7:].split('=',1)
            parts = shlex.split(value)
            assert len(parts) == 1, (key,value)
            result[key] = parts[0]
    return result

def toml_text(data):
    def scalar(v):
        if isinstance(v,bool): return str(v).lower()
        if isinstance(v,str): return json.dumps(v)
        if isinstance(v,(int,float)): return repr(v)
        if isinstance(v,list): return '[' + ', '.join(scalar(x) for x in v) + ']'
        raise TypeError(type(v))
    lines = []
    def emit(d,prefix=''):
        for k,v in d.items():
            if not isinstance(v,dict) and not (isinstance(v,list) and v and isinstance(v[0],dict)):
                lines.append(k+' = '+scalar(v))
        for k,v in d.items():
            name = prefix+'.'+k if prefix else k
            if isinstance(v,dict):
                lines.extend(['','['+name+']']); emit(v,name)
            elif isinstance(v,list) and v and isinstance(v[0],dict):
                for item in v:
                    lines.extend(['','[['+name+']]']); emit(item,name)
    emit(data)
    out = '\n'.join(lines)+'\n'
    assert tomllib.loads(out) == data
    return out

def source_for(case,workflow,angle):
    if workflow == 'hybrid':
        if case == 's03_c022_case12':
            return BASE / ('hybrid_pc30_production_kr_s03_c022_case12_20260910_v1/reservoir' if angle == 30
                else 'hybrid_pc60_production_kr_s03_c022_case12_20260911_v1/reservoir_preflightfix1')
        return BASE / f'hybrid_low_high_wrr30_60_20260911_v1/{case}/wrr{angle}/reservoir'
    if case == 's03_c022_case12' and angle == 30:
        return BASE / 'reservoir_comparison_s03_c022_case12_20260910_v1'
    return MATRIX / f'variants/{case}_wrr{angle}_physical_prebreakthrough'

def replace_required(text,old,new,count=None):
    n=text.count(old)
    assert n>0 and (count is None or count==n), (old,n,count)
    return text.replace(old,new)

def seal(dest):
    files = [p for p in dest.iterdir() if p.is_file() and p.name!='STUDY_SHA256.txt']
    files += list((dest/'source').glob('*'))
    lines = [f'{sha(p)}  {p.relative_to(dest)}' for p in sorted(files) if p.is_file()]
    for p in [ROOT/'hysteresis_validation.jl',ROOT/'reference/advanced.jl',ROOT/'reference/production_qoi.jl']:
        lines.append(f'{sha(p)}  {p}')
    (dest/'STUDY_SHA256.txt').write_text('\n'.join(lines)+'\n')

def prepare():
    commit = subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip()
    assert commit != BASE_COMMIT
    assert not subprocess.check_output(['git','-C',str(RUNTIME),'status','--porcelain','--untracked-files=no'],text=True).strip()
    assert sha(RUNTIME/'Manifest.toml') == sha(ORIGINAL_RUNTIME/'Manifest.toml')
    original_diff = subprocess.check_output(['git','-C',str(RUNTIME),'diff','--name-only',BASE_COMMIT,commit],text=True).splitlines()
    assert sorted(original_diff)==sorted(['src/variables/relperm/advanced.jl','src/input_simulation/mrst_input.jl','src/input_simulation/production_qoi.jl'])
    template_env = readenv(TEMPLATE/'locked.env')
    template_workflow = tomllib.loads((TEMPLATE/'campaign.toml').read_text())['workflow']
    records=[]
    for risk,case,task in CASES:
        for workflow,angle in [('production',30),('mip',30),('mip',60),('hybrid',30),('hybrid',60)]:
            original = source_for(case,workflow,angle)
            name=f'{risk}_{workflow}_wrr{angle}_fault_nohys'
            dest=ROOT/'variants'/name
            assert not (dest/'results').exists(), f'Refuse to edit launched case {dest}'
            env=readenv(original/'locked.env')
            oldmanifest=tomllib.loads((original/'campaign.toml').read_text())
            manifest=copy.deepcopy(oldmanifest)
            assert len(manifest['cases'])==1
            entry=manifest['cases'][0]
            assert entry['case_key']==case and entry['task']==task
            source_specific=Path(entry['specific_path'])
            assert sha(source_specific)==entry['specific_sha256']
            if workflow=='production':
                source_specific=Path(env['BASELINE_SPECIFIC_MAT'])
                entry.update(specific_path=str(source_specific),specific_sha256=sha(source_specific),
                    specific_bytes=source_specific.stat().st_size)
            assert sha(Path(manifest['common']['path']))==manifest['common']['sha256']
            manifest.update(campaign_id=name+'_20260911',archive_root=str(dest),jutuldarcy_commit=commit,
                production_promotion=False,restart_reuse=False)
            manifest['workflow']=copy.deepcopy(template_workflow)
            manifest['correction']={'schema':'fault_only_hysteresis_15case_v1',
                'base_runtime_commit':BASE_COMMIT,'runtime_commit':commit,'runtime_path':str(RUNTIME),
                'source_campaign':str(original/'campaign.toml'),'source_campaign_sha256':sha(original/'campaign.toml'),
                'risk':risk,'closure':workflow,'wrr_angle_deg':angle,
                'fault_hysteresis':'explicit_no_hysteresis','host_hysteresis':'unchanged_production_Killough',
                'pc_convention':'unchanged_original_control' if workflow=='production' else 'physical_prebreakthrough',
                'start_from_time_zero':True,'all_210_report_states_retained':True}
            dest.mkdir(parents=True,exist_ok=True)
            source=dest/'source';source.mkdir(exist_ok=True)
            # Preserve each proven input-pair audit, including scenario-specific provenance.
            for p in (original/'source').iterdir():
                if p.is_file(): shutil.copy2(p,source/p.name)
            # Use the movie-ready shared scheduler/retention policy for every workflow.
            for name0 in [PREFIX+'_full.sbatch',PREFIX+'_preflight.sbatch',PREFIX+'_common.sh',
                          PREFIX+'_final_check.jl','gom_step62_production_final_check.jl',PREFIX+'_runtime_diagnostics.jl']:
                shutil.copy2(TEMPLATE/'source'/name0,source/name0)
            if workflow=='production':
                for name0 in [PREFIX+'_contract.jl',PREFIX+'_preflight.jl']:
                    shutil.copy2(ORIGINAL_RUNTIME/'scripts/engaging'/name0,source/name0)
                shutil.copy2(ROOT/'production_pair_audit.jl',source/'pair_input_audit.jl')
            for p in source.iterdir():
                if p.suffix not in ('.jl','.sh','.sbatch'): continue
                s=p.read_text().replace('\r\n','\n')
                s=s.replace('reservoir_only_fault_drainage_duplicate','reservoir_only_fault_explicit_no_hysteresis')
                s=s.replace('fault_hysteresis=drainage_equivalent','fault_hysteresis=explicit_no_hysteresis')
                if p.name==PREFIX+'_preflight.jl':
                    if workflow=='production':
                        s=replace_required(s,'mkpath(result_dir)','mkpath(result_dir)\ninclude(joinpath(@__DIR__, "pair_input_audit.jl"))\naudit_candidate_pair(specific_path, result_dir)',1)
                    s=replace_required(s,'import MAT','import MAT\ninclude(joinpath(ENV["CORRECTED_STUDY_ROOT"], "hysteresis_validation.jl"))',1)
                    s=replace_required(s,'summary_path = joinpath(result_dir, "preflight_summary.txt")',
                        'CorrectedFaultValidation.validate!(setup, result_dir)\nsummary_path = joinpath(result_dir, "preflight_summary.txt")',1)
                if p.suffix=='.sbatch':
                    s=re.sub(r'#SBATCH --array=\d+',f'#SBATCH --array={task}',s)
                    if workflow=='production':
                        s=s.replace('pc_plateau_adjusted_tables=8','pc_plateau_adjusted_tables=530')
                        s=s.replace('pc_plateau_already_plateaued_tables=522','pc_plateau_already_plateaued_tables=0')
                    if p.name==PREFIX+'_full.sbatch':
                        needle='cp -- "$preflight_summary" "$case_dir/PREFLIGHT_PC_TABLE_CONTRACT.txt"'
                        replacement='grep -Fxq \'fault_hysteresis=explicit_no_hysteresis\' "$preflight_summary"\n'
                        replacement+='test -s "$(dirname "$preflight_summary")/fault_hysteresis_validation.toml"\n'
                        replacement+='cp -- "$(dirname "$preflight_summary")/fault_hysteresis_validation.toml" "$case_dir/"\n'+needle
                        s=replace_required(s,needle,replacement,1)
                        s=s.replace('retained_restart_steps=51,78,110,210','retained_restart_steps=1:210')
                        s=s.replace('"hysteresis_active=true" \\', '"hysteresis_active=true" \\\n    "fault_hysteresis=explicit_no_hysteresis" \\')
                        s=s.replace('restart/jutul_51.jld2 restart/jutul_78.jld2 \\\n        restart/jutul_110.jld2 restart/jutul_210.jld2', 'restart/jutul_*.jld2')
                if p.name==PREFIX+'_common.sh':
                    if workflow=='production':
                        s=s.replace('pc_entry_expected_adjusted_tables=8','pc_entry_expected_adjusted_tables=530')
                        s=s.replace('pc_entry_expected_already_plateaued_tables=522','pc_entry_expected_already_plateaued_tables=0')
                    s+='\n# Fault-only correction is recorded separately from unchanged host Pc conventions.\n'
                p.write_text(s)
            (dest/'campaign.toml').write_text(toml_text(manifest))
            manifest_sha=sha(dest/'campaign.toml')
            (dest/'campaign.toml.sha256').write_text(manifest_sha+'  campaign.toml\n')
            env.update(GOM_PRODUCTION_CAMPAIGN_ID=manifest['campaign_id'],
                GOM_PRODUCTION_MANIFEST_PATH=str(dest/'campaign.toml'),
                GOM_PRODUCTION_MANIFEST_COMPANION_PATH=str(dest/'campaign.toml.sha256'),
                GOM_PRODUCTION_MANIFEST_SHA256=manifest_sha,
                GOM_PRODUCTION_ARCHIVE_ROOT=str(dest),GOM_PRODUCTION_JUTULDARCY_COMMIT=commit,
                GOM_PRODUCTION_SPECIFIC_MAT=str(source_specific),GOM_PRODUCTION_SPECIFIC_SHA256=entry['specific_sha256'],
                GOM_PRODUCTION_SPECIFIC_BYTES=str(entry['specific_bytes']),
                GOM_PRODUCTION_RETAIN_YEARS=template_env['GOM_PRODUCTION_RETAIN_YEARS'],
                GOM_PRODUCTION_ARCHIVE_COMPACT_ATOMIC_ROWS='false',
                GOM_PRODUCTION_ARCHIVE_REMOVE_VERIFIED_SCRATCH='false',
                CORRECTED_STUDY_ROOT=str(ROOT),CORRECTED_CLOSURE=workflow,CORRECTED_WRR_ANGLE=str(angle))
            (dest/'locked.env').write_text(''.join(f'export {k}={shlex.quote(v)}\n' for k,v in env.items()))
            guard=(ROOT/'guard.sh.in').read_text().replace('@ROOT@',str(dest)).replace('@TASK@',str(task)).replace('@CASE@',case)
            (source/'candidate_guard.sh').write_text(guard)
            # Keep every transitive source input checksum; inputs remain at their original read-only locations.
            pins={str(source_specific):entry['specific_sha256'],manifest['common']['path']:manifest['common']['sha256']}
            for line in (original/'INPUTS_SHA256.txt').read_text().splitlines():
                digest,path=line.split(maxsplit=1);path=path.lstrip('*')
                path=str((original/path).resolve()) if not Path(path).is_absolute() else path
                assert sha(path)==digest,(path,'source checksum mismatch')
                pins[path]=digest
            (dest/'INPUTS_SHA256.txt').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(pins.items())))
            (dest/'provenance.json').write_text(json.dumps(manifest['correction'],indent=2)+'\n')
            for p in source.iterdir():
                if p.suffix in ('.sh','.sbatch'):
                    subprocess.run(['bash','-n',str(p)],check=True)
            seal(dest)
            records.append(dict(name=name,risk=risk,case=case,task=task,workflow=workflow,angle=angle,
                root=str(dest),source=str(original),specific=str(source_specific),specific_sha256=entry['specific_sha256'],
                runtime_commit=commit))
    assert len(records)==15
    (ROOT/'matrix.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))

def submit(unit_job):
    records=json.loads((ROOT/'matrix.json').read_text())
    ledger=ROOT/'submissions.json'
    assert not ledger.exists(),'Submission ledger already exists; no duplicate jobs allowed.'
    submitted=[]
    ledger.write_text('[]\n')
    for row in records:
        root=Path(row['root']);task=str(row['task'])
        def launch(kind,dep,pf=''):
            cmd=['sbatch','--parsable','--partition=mit_normal','--account=mit_amf_advanced_cpu',
                 '--qos=mit_amf_advanced_cpu','--cpus-per-task=8','--mem='+('24G' if kind=='preflight' else '18G'),
                 '--time='+('04:00:00' if kind=='preflight' else '3-00:00:00'),
                 '--no-requeue','--array='+task,
                 '--job-name=h15_'+row['risk'][:3]+'_'+row['workflow'][:3]+'_'+str(row['angle'])+'_'+kind[:2],
                 '--output='+str(ROOT/'logs'/(row['name']+'_'+kind+'-%A_%a.out')),
                 '--error='+str(ROOT/'logs'/(row['name']+'_'+kind+'-%A_%a.err')),
                 str(ROOT/'launch.sh'),str(root),kind,pf]
            if dep:
                cmd.insert(1,'--dependency=afterok:'+dep)
            job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
            assert job.isdigit(),job
            return job
        entry=dict(row)
        entry['preflight_job']=launch('preflight','')
        submitted.append(entry);ledger.write_text(json.dumps(submitted,indent=2)+'\n')
        entry['full_job']=launch('full',unit_job+':'+entry['preflight_job'],entry['preflight_job'])
        ledger.write_text(json.dumps(submitted,indent=2)+'\n')
        print(row['name'],entry['preflight_job'],entry['full_job'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','submit']);ap.add_argument('--unit-job')
    args=ap.parse_args()
    if args.action=='prepare': prepare()
    else:
        assert args.unit_job and args.unit_job.isdigit()
        submit(args.unit_job)
