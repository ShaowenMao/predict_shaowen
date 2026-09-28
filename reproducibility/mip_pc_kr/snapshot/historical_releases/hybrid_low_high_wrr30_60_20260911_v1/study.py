"""Isolated candidate Pc / recomputed production dynamic-Kr comparison."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import tomllib

SOURCE = Path(__file__).resolve().parent
BASE = Path('/orcd/data/juanes/001/shaowen/predict_shaowen')
CAND = BASE/'candidates/cell_pc_mip_kr_v1_20260908'
SUITE = CAND/'hybrid_low_high_wrr30_60_20260911_v1'
CASE = os.environ.get('HYBRID_CASE', 's01_c018_case07')
ANGLE = int(os.environ.get('HYBRID_ANGLE', '30'))
CASE_IDS = {'s01_c018_case07':262, 's03_c003_case08':848}
assert CASE in CASE_IDS and ANGLE in (30,60)
GEO = CASE.split('_case')[0]
REALIZATION = int(CASE.split('_case')[1])
TASK = CASE_IDS[CASE]
OUT = SUITE/CASE/f'wrr{ANGLE}'
NEW = CAND/'comparative_3case_wrr30_60_pc_conventions_20260910_v1'
FROZEN = CAND/'releases/fullfault_julia_1bc6badb9d4d'
PC_MODULE = FROZEN/'vendor/pipeline/vendor/pc/src/FaultCorePc.jl'
P30 = CAND/'full_fault_julia_s03_c022_case12_20260909_v1'
P60 = CAND/'full_fault_wrr60_s03_c022_case12_20260910_v1'
EXPORT30 = CAND/'reservoir_export_s03_c022_case12_20260910_v1'
EXPORT = OUT/'pc_export'
MATCH30 = CAND/'hybrid_pc30_production_kr_s03_c022_case12_20260910_v1'
PAIR = CAND/'reservoir_comparison_s03_c022_case12_20260910_v1'
PROD = BASE/'production_runs/independent_full_fault_v1_phase1_20260809_v1'
RUNTIME = BASE/'orchestration/independent_full_fault_v1_04927f5/runtime_repo'
PHYSICS = RUNTIME.parent/'physics_repo'
DATA = BASE/'production_freezes/collapsed_cell_union_20260722_v7/inputs/predict'
MRST = Path('/orcd/pool/006/shaowen/predict_shaowen/software/mrst-2025a/SINTEF-AppliedCompSci-MRST-75749fa')
ZIP = MRST.parents[1]/'upscaling.zip'
JUTUL = Path('/home/shaowen/orcd/scratch/gom_grid/code/immutable/JutulDarcy_ed81af8e')
DRIVER = 'run_kr_upscaling_dyn_median_examples_full87.m'
DRIVER_HASH = 'fda0cc735d882fdfb0cfab228004cedc0a4ccabaccce478c888e8414ea20564d'
MOVIE = CAND/'releases/comparative_movie_output_20260910_v1'
MOVIE_DATA = CAND/'comparative_3case_wrr30_60_pc_conventions_20260910_v1/movie_output_policy_v1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(x, f, indent=2, allow_nan=False)


def rows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csvwrite(p, records):
    with Path(p).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def hashes(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.name != 'source_lock.json'}


def check_files(values):
    for p, h in values.items():
        assert sha(p) == h, ('Changed immutable input', p)


def verify():
    assert SOURCE == Path(os.environ['HYBRID_SOURCE']).resolve()
    assert Path(os.environ['HYBRID_ROOT']).resolve() == OUT
    assert CAND in OUT.parents and 'scratch' not in OUT.parts
    assert hashes(SOURCE) == read(SOURCE/'source_lock.json')['files']
    if (OUT/'prepared.json').exists():
        p = read(OUT/'prepared.json')
        check_files(p['protected'])
        check_files({str(OUT/k): v for k, v in p['staged_files'].items()})


def seal():
    assert not (SOURCE/'source_lock.json').exists()
    write(SOURCE/'source_lock.json',dict(files=hashes(SOURCE)))


def command(folder, args, env=None):
    folder.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    print('START', folder, flush=True)
    with (folder/'process.log').open('x') as f:
        rc = subprocess.run(['/usr/bin/time', '-v', '-o', str(folder/'resources.txt')]+args,
                            cwd=folder, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    write(folder/'receipt.json', dict(returncode=rc, seconds=time.monotonic()-start, args=args))
    assert rc == 0, f'{folder}: return code {rc}'


def mq(p):
    return "'"+str(p).replace("'", "''")+"'"


def matlab(folder, expr, env=None):
    command(folder, ['matlab', '-batch', f'addpath({mq(SOURCE)});'+expr], env)


def medoid(group):
    vals = sorted(float(x['EffectiveSwi']) for x in group)
    assert len(vals) == 87 and all(math.isfinite(x) for x in vals)
    target = vals[43]
    return min(group, key=lambda r: (abs(float(r['EffectiveSwi'])-target),
                                    int(r['ReplaySourceRow']), int(r['SliceIndex'])))


def reference_deck(text, clay):
    """Replace only the clay SGOF record, preserving sand, PVT and all other text."""
    import numpy as np
    match = re.search(r'(?m)^SGOF\s*$', text)
    assert match is not None
    first = text.index('/', match.end())
    second = text.index('/', first+1)
    old = np.loadtxt(text[first+1:second].splitlines())
    assert old.shape == clay.shape == (32, 4)
    assert old[-1, 0] == .7 and clay[-1, 0] == .8
    assert clay[0, 1] == 0 and clay[0, 2] == 1 and clay[-1, 1] == 1 and clay[-1, 2] == 0
    swstar = 1-clay[:, 0]/.8
    np.testing.assert_allclose(clay[:, 1], (1-swstar)**2.5, rtol=2e-13, atol=1e-15)
    np.testing.assert_allclose(clay[:, 2], swstar**3.5, rtol=2e-13, atol=1e-15)
    b = math.log(old[-1, 3]/old[1, 3])/(old[-1, 0]-old[1, 0])
    a = old[1, 3]/math.exp(b*old[1, 0])
    np.testing.assert_allclose(clay[1:, 3]/1e5, a*math.cos(math.pi/3)/math.cos(math.pi/6)*np.exp(b*clay[1:, 0]), rtol=2e-12)
    deck = clay.copy(); deck[:, 3] /= 1e5  # Reference archive Pa; METRIC deck bar.
    body = '\n'+'\n'.join('    '+' '.join(format(v,'.17g') for v in row) for row in deck)+'\n'
    result = text[:first+1]+body+text[second:]
    assert result[:first+1] == text[:first+1] and result.endswith(text[second:])
    return result


def pc_nodes(pressure, saturation, entry):
    """The existing native-node Pc export policy, independent of diagnostic Kr."""
    import numpy as np
    p=np.asarray(pressure); sg=np.asarray(saturation)
    assert len(p)==len(sg)==100 and p[0]==sg[0]==0
    assert np.isfinite(p).all() and np.isfinite(sg).all()
    assert np.all(np.diff(p)>0) and np.all(np.diff(sg)>=0)
    pos=np.flatnonzero(sg>0); assert len(pos)>0
    first=pos[0]; assert p[first-1]<=entry<=p[first]
    repeated=np.flatnonzero(np.diff(sg)==0)+1
    assert all(sg[i]==0 or sg[i]==sg[-1] for i in repeated), 'Interior vertical Pc branch requires review'
    keep=np.r_[0,np.flatnonzero(np.diff(sg)>0)+1]
    t=np.column_stack([sg[keep],p[keep]]); t[0,1]=entry
    assert np.all(np.diff(t[:,0])>1e-12) and np.all(np.diff(t[:,1])>=0)
    return t


def case_manifest():
    m=read(NEW/'inputs/manifest.json')
    c=m['cases'][CASE]
    assert c['production']['task']==TASK and c['production']['geology_id']==GEO
    assert c['production']['realization_id']==REALIZATION
    return m,c


def stage_pc():
    """Reuse accepted Pc; recover only missing Pc with the same frozen MIP engine."""
    import numpy as np
    from scipy.io import savemat, loadmat
    m,c=case_manifest()
    frozen=read(FROZEN/'source_lock.json')
    check_files({str(FROZEN/k):v for k,v in frozen['files'].items()})
    EXPORT.mkdir()
    protected={str(NEW/'inputs/manifest.json'):sha(NEW/'inputs/manifest.json')}
    protected.update({str(FROZEN/k):v for k,v in frozen['files'].items()})
    protected[str(FROZEN/'source_lock.json')]=sha(FROZEN/'source_lock.json')
    jobs=[]; references=set(); entries=[]
    for r in c['records']:
        slot=int(r['slot']); a=r['assignment']
        assert slot==(int(a['window'][4:])-1)*87+int(a['slice_index'])
        receipt=NEW/f'datasets/{CASE}/wrr{ANGLE}/accepted/slot{slot:03}.json'
        accepted=receipt.exists()
        if accepted:
            rc=read(receipt)
            assert rc['numerical_checks_passed'] and rc['calculation_key']==r['calculation_key']
            assert rc['case']==CASE and rc['angle']==ANGLE and rc['slot']==slot
            assert rc['input_manifest_sha256']==sha(NEW/'inputs/manifest.json')
            attempt=Path(rc['attempt'])
            for rel in ('numerical/planned/raw.bin','numerical/coupled/pc_kr_raw.csv',
                        'numerical/coupled/summary.toml','numerical/coupled/completion.toml'):
                p=attempt/rel; assert sha(p)==rc['outputs'][rel];protected[str(p)]=sha(p)
            raw=attempt/'numerical/planned/raw.bin'
            curve=attempt/'numerical/coupled/pc_kr_raw.csv'
            summary=tomllib.loads((attempt/'numerical/coupled/summary.toml').read_text())
            assert summary['raw_input_sha256']==sha(raw) and summary['points']==100
            protected[str(receipt)]=sha(receipt)
        else:
            # A failed Kr solve is not a failed Pc calculation. Read its immutable
            # planned input, but never accept a partial coupled curve.
            options=sorted((NEW/f'attempts/{CASE}').glob(f'slot{slot:03}_*/wrr{ANGLE}/numerical/planned/raw.bin'))
            assert options, ('Missing planned core input',CASE,ANGLE,slot)
            raw=options[-1]; attempt=raw.parents[2]
            prov=attempt.parent/'provenance.json'
            assert read(prov)['record']==r
            protected[str(prov)]=sha(prov);protected[str(raw)]=sha(raw)
            curve=Path('-')
        if ANGLE==60:
            scenario=attempt/'numerical/planned/scenario_audit.toml'
            audit=tomllib.loads(scenario.read_text())
            assert audit['passed'] and audit['clay_swi']==.2 and audit['clay_nw']==3.5 and audit['clay_ng']==2.5
            rc30=read(NEW/f'datasets/{CASE}/wrr30/accepted/slot{slot:03}.json')
            parent=Path(rc30['attempt'])/'numerical/planned/raw.bin'
            assert audit['raw_parent_sha256']==sha(parent)
            ref=attempt/'numerical/planned/clay_reference_60.csv'
            references.add(sha(ref));protected[str(ref)]=sha(ref);protected[str(scenario)]=sha(scenario)
        jobs.append(dict(slot=slot,raw=str(raw),curve=str(curve),accepted=int(accepted),
                         output=str(EXPORT/f'pc_slot{slot:03}.csv')))
    if ANGLE==60:
        assert references=={'48a010171241eb9110d287d6bfbee13427d635985c3a1ea82eed903970028d63'}
        shutil.copy2(ref,EXPORT/'clay_reference_60.csv')
    csvwrite(EXPORT/'jobs.csv',jobs)
    cmd='module purge\nmodule load deprecated-modules gcc/12.2.0-x86_64 julia/1.12.6\n'
    cmd+='export JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 JULIA_LOAD_PATH="@:@stdlib"\n'
    cmd+='exec '+shlex.join(['julia','--startup-file=no','--math-mode=ieee','--check-bounds=yes',
          str(SOURCE/'stage_pc.jl'),str(PC_MODULE),str(EXPORT/'jobs.csv'),str(EXPORT/'entries.csv')])
    command(OUT/'pc_stage_process',['bash','-lc',cmd])
    meta=rows(EXPORT/'entries.csv')
    assert len(meta)==522 and [int(x['slot']) for x in meta]==list(range(1,523))
    tables=np.empty((6,87),dtype=object)
    for j,e in zip(jobs,meta):
        values=np.loadtxt(j['output'],delimiter=',',skiprows=1)
        table=pc_nodes(values[:,0],values[:,1],float(e['entry_pressure_pa']))
        slot=j['slot'];w=(slot-1)//87;s=(slot-1)%87
        tables[w,s]=table
        entries.append(dict(slot=slot,entry_pressure_pa=float(e['entry_pressure_pa'])))
    write(EXPORT/'entry_metadata.json',entries)
    file=EXPORT/f'pc_tables_{CASE}.mat'
    savemat(file,dict(pcTables=tables,tableSlotMap=np.arange(1,523).reshape(6,87),
        entryPressurePa=np.array([r['entry_pressure_pa'] for r in entries]).reshape(6,87),
        rawSourceSha256=sha(NEW/'inputs/manifest.json'),entryMetadataSha256=sha(EXPORT/'entry_metadata.json'),
        schema='candidate_hybrid_pc_only_v1',wrrContactAngleDeg=ANGLE,caseKey=CASE,
        productionPromotion=False),do_compression=True)
    reread=loadmat(file)['pcTables']
    assert all(np.array_equal(tables[w,s],reread[w,s]) for w in range(6) for s in range(87))
    write(EXPORT/'audit.json',dict(cores=522,all_structural_checks_passed=True,
          angle=ANGLE,pc_convention='physical_prebreakthrough',
          accepted_curves_reused=sum(j['accepted'] for j in jobs),
          pc_only_recoveries=[j['slot'] for j in jobs if not j['accepted']],
          production_promotion=False))
    write(EXPORT/'source_receipts.json',protected)
    print('STAGED',CASE,ANGLE,read(EXPORT/'audit.json'),flush=True)


def prepare():
    from scipy.io import loadmat
    import numpy as np
    assert not (OUT/'prepared.json').exists()
    assert shutil.disk_usage(OUT).free > 200*1024**3
    assert read(MATCH30/'qualified.json')['passed']
    stage_pc()
    cin = PROD/f'case_inputs/cases/{GEO}/case{REALIZATION:02}/inputs'
    cres = PROD/f'case_results/cases/{GEO}/case{REALIZATION:02}'
    original = rows(cres/f'kr/tables/kr_curve_summary_{GEO}_cases_{REALIZATION:02}_dyn_swi_medoid.csv')
    assert len(original) == 6
    for r in original:
        assert (r['KrMode'], r['OneDMatchMethod'], r['OneDAdSolver']) == ('dyn', 'ad', 'robust')
        assert float(r['CoreyExponentStep']) == .2 and r['GridMode'] == 'full3d'
    assert sha(RUNTIME/'examples/pc_upscaling_pilot'/DRIVER) == DRIVER_HASH
    staging = OUT/'runtime'
    (staging/'examples/pc_upscaling_pilot').mkdir(parents=True)
    # Freeze the archived production orchestration, not the newer working copy.
    for source in (RUNTIME/'examples').glob('*.m'):
        shutil.copy2(source, staging/'examples'/source.name)
    for source in (RUNTIME/'examples/pc_upscaling_pilot').glob('*.m'):
        shutil.copy2(source, staging/'examples/pc_upscaling_pilot'/source.name)
    assert sha(staging/'examples/pc_upscaling_pilot'/DRIVER) == DRIVER_HASH
    driver_path=staging/'examples/pc_upscaling_pilot'/DRIVER
    old_driver=driver_path.read_text()
    old_deck="cfg.originalDeckFile = fullfile(cfg.upscalingRoot, 'eclipse_data_files', ...\n    'gom_forUps_theta30_PVDO_incompRock.DATA');"
    assert old_driver.count(old_deck)==old_driver.count('krOpt.contactAngleDeg = 30;')==1
    new_driver=old_driver.replace(old_deck,'cfg.originalDeckFile = getenv("KR_DYN_REFERENCE_DECK_FILE");').replace('krOpt.contactAngleDeg = 30;',f'krOpt.contactAngleDeg = {ANGLE};')
    driver_path.write_text(new_driver)
    inputs = OUT/'inputs'
    inputs.mkdir()
    base_deck=MATCH30/'windows/w1/upscaling/eclipse_data_files/gom_forUps_theta30_PVDO_incompRock.DATA'
    deck=inputs/f'gom_forUps_theta{ANGLE}_PVDO_incompRock.DATA'
    if ANGLE==60:
        clay=np.loadtxt(EXPORT/'clay_reference_60.csv',delimiter=',')
        with deck.open('x') as f:f.write(reference_deck(base_deck.read_text(),clay))
        matlab(OUT/'reference_check',f'check_wettability({mq(base_deck)},{mq(deck)},{mq(EXPORT/"clay_reference_60.csv")},{mq(MRST)});')
    else:
        shutil.copy2(base_deck,deck);assert sha(deck)==sha(base_deck)
    write(inputs/'wettability_contract.json',dict(angle=ANGLE,
          clay_swi=.2 if ANGLE==60 else .3,clay_sgmax=.8 if ANGLE==60 else .7,
          sand_unchanged=True,PVT_unchanged=True,deck_sha256=sha(deck),
          archived_driver_sha256=DRIVER_HASH,scenario_driver_sha256=sha(driver_path),
          only_driver_changes=['reference deck path','scenario contact angle']))
    _,m=case_manifest()
    tables = loadmat(EXPORT/f'pc_tables_{CASE}.mat')['pcTables']
    assert tables.shape == (6, 87)
    template_path = cin/f'replay_summary_template_{CASE}.csv'
    template = rows(template_path)
    assert len(template) == len(m['records']) == 522
    lookup = {(r['Window'], int(r['SliceIndex'])): r for r in template}
    summary, native, assignments = [], [], {}
    for r in m['records']:
        a = r['assignment']; old = r['production_summary']
        w, s = int(a['window'][4:]), int(a['slice_index'])
        tr = lookup[(a['window'], s)]
        assert tr['TaskId'] == a['task_id'] == old['TaskId']
        assert int(tr['SelectedSampleIndex']) == int(a['selected_sample_index'])
        assert int(tr['ReplaySeed']) == int(a['exact_replay_seed'])
        t = tables[w-1, s-1]
        assert len(t) >= 2 and t[0,0] == 0 and np.all(np.diff(t[:,0]) > 0)
        new = dict(old)
        new.update(BulkSgMax=repr(float(t[-1,0])), EffectiveSwi=repr(float(1-t[-1,0])),
                   PcMinPa=repr(float(t[0,1])), PcMaxPa=repr(float(t[-1,1])),
                   RawNumPoints=str(len(t)), PcSource=f'candidate_WRR{ANGLE}_physical_prebreakthrough')
        for v in (20, 50, 65):
            new[f'PcAtSg{v}Pa'] = repr(float(np.interp(v/100, t[:,0], t[:,1])))
        # Do not mislabel old production breakthrough/scaling diagnostics as new.
        for k in ('PercolationPcPa','PercolationSg','ScalingPermComponent','IpT'):
            new[k] = ''
        summary.append(new)
        for sg, pc in t:
            native.append(dict(ReplaySourceRow=tr['SourceRow'], GasSaturation=repr(float(sg)),
                               PcPa=repr(float(pc))))
        a = dict(a); a['case_source_row'] = tr['SourceRow']
        assignments[(a['window'],s)] = a
    assert {int(r['ReplaySourceRow']) for r in summary} == set(range(1,523))
    csvwrite(inputs/'pc_summary.csv', summary)
    csvwrite(inputs/'pc_native.csv', native)
    csvwrite(inputs/'replay_template.csv', template)
    chosen = [medoid([r for r in summary if r['Window'] == f'famp{w}']) for w in range(1,7)]
    selected = [assignments[(r['Window'],int(r['SliceIndex']))] for r in chosen]
    csvwrite(inputs/'selected.csv', selected)
    for w, x in enumerate(selected,1):
        csvwrite(inputs/f'selection_w{w}.csv', [x])
    csvwrite(inputs/'original_dynamic_summary.csv', original)
    matlab(OUT/'selection_check', f'addpath({mq(staging/"examples/pc_upscaling_pilot")});check_selection({mq(OUT)},{REALIZATION});')
    protected=read(EXPORT/'source_receipts.json')
    _,cm=case_manifest()
    baseline=Path(cm['production']['specific_path'])
    assert sha(baseline)==cm['production']['specific_sha256']
    for p in [EXPORT/f'pc_tables_{CASE}.mat',baseline,template_path,cres/'case.done.json',
              RUNTIME/'examples/pc_upscaling_pilot'/DRIVER,ZIP,PAIR/'campaign.toml',
              PAIR/'locked.env',PAIR/'STUDY_SHA256.txt',base_deck,
              EXPORT30/'smoke_validation_v3.jl',MATCH30/'qualified.json',
              EXPORT/'audit.json',PC_MODULE]:
        protected[str(p)]=sha(p)
    files = {str(p.relative_to(OUT)):sha(p) for d in (inputs,staging) for p in d.rglob('*') if p.is_file()}
    write(OUT/'prepared.json', dict(passed=True, protected=protected, staged_files=files,
        selected=[dict(window=x['window'],slice=int(x['slice_index']),sample=int(x['selected_sample_index']),
                       seed=int(x['exact_replay_seed']),source_row=int(x['case_source_row'])) for x in selected],
        production_dynamic_driver_sha256=DRIVER_HASH, scenario_driver_sha256=sha(driver_path),
        production_dynamic_algorithm_unchanged=True, wettability_scenario=ANGLE,
        candidate_Pc_unchanged=True, reservoir_submitted=False))
    print('PREPARED six new medoids:', [(x['window'],x['slice_index']) for x in selected], flush=True)


def window(w):
    assert 1 <= w <= 6
    root = OUT/f'windows/w{w}'
    root.mkdir(parents=True, exist_ok=False)
    local = OUT/'runtime/examples/pc_upscaling_pilot'
    replay = root/'replay'
    matlab(root/'replay_process', f'addpath({mq(local)});prepare_production_replay_batch('
           f'{mq(OUT/"inputs"/f"selection_w{w}.csv")},{mq(replay)},{mq(DATA)},{mq(PHYSICS)},{mq(MRST)},0.005);')
    rr = rows(replay/'tables/replay_summary_context.csv')
    selected = rows(OUT/'inputs/selected.csv')[w-1]
    assert len(rr) == 1 and rr[0]['TaskId'] == selected['task_id']
    r = rr[0]
    assert r['VerificationStatus'] == 'matched' and float(r['MaxAbsLog10Diff']) <= .005
    assert int(r['SelectedSampleIndex']) == int(selected['selected_sample_index'])
    assert int(r['ReplaySeed']) == int(selected['exact_replay_seed'])
    template = rows(OUT/'inputs/replay_template.csv')
    for t in template:
        t['OutputFile'] = 'NOTREPLAYED'
    target = template[int(selected['case_source_row'])-1]
    assert target['TaskId'] == selected['task_id']
    target.update(OutputFile=r['OutputFile'],VerificationStatus=r['VerificationStatus'],MaxAbsLog10Diff=r['MaxAbsLog10Diff'])
    csvwrite(root/'replay_case.csv', template)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('KR_DYN_'):
            del env[key]
    settings = dict(GEOLOGY_ID=GEO, CASE_IDS=str(REALIZATION), REPLAY_ROOT=str(replay),
        REFERENCE_DECK_FILE=str(OUT/f'inputs/gom_forUps_theta{ANGLE}_PVDO_incompRock.DATA'),
        REPLAY_SUMMARY_CSV=str(root/'replay_case.csv'), OUTPUT_ROOT=str(root/'kr'),
        SELECTION_MODE='swi_medoid', PC_PRESTEP_MODE='precomputed',
        PRECOMPUTED_PC_SUMMARY_CSV=str(OUT/'inputs/pc_summary.csv'),
        PRECOMPUTED_PC_CURVE_CSV=str(OUT/'inputs/pc_native.csv'),
        PRECOMPUTED_PC_NATIVE_CURVE_CSV=str(OUT/'inputs/pc_native.csv'),
        UPSCALING_ROOT=str(root/'upscaling'), ONLY_ROWS=str(w), USE_PARALLEL='0',
        MAX_ROWS='', SMOKE_CARTDIMS='', TIMESTEP_MODE='paper', COREY_STEP='0.2',
        **{'1D_METHOD':'ad','1D_AD_SOLVER':'robust','1D_LINEAR_SOLVER':'amgcl_auto',
           '3D_NUM_THREADS':'1'}, LINEAR_SOLVER='amgcl_require', EXPORT_RESERVOIR_READY='0',
        DISABLE_RUN_MAT_CACHE='1')
    env.update({'KR_DYN_'+k:v for k,v in settings.items()})
    env.update(MRST_ROOT=str(MRST), UPSCALING_ZIP=str(ZIP))
    write(root/'settings.json', settings)
    # Prevent inner 1-D parfor from silently allocating an unrequested process pool.
    expr = "s=parallel.Settings;s.Pool.AutoCreate=false;maxNumCompThreads(1);"+f'run({mq(local/DRIVER)});'
    matlab(root/'dynamic_process', expr, env)
    report = rows(root/f'kr/tables/kr_curve_summary_{GEO}_cases_{REALIZATION:02}_dyn_swi_medoid.csv')
    assert len(report) == 1
    r = report[0]
    for k in ('PcMaxSg','IrreducibleWaterSaturation','HistoryMatchError','Dynamic3DSeconds','Dynamic1DMatchSeconds'):
        assert math.isfinite(float(r[k])), k
    expected = read(OUT/'prepared.json')['selected'][w-1]
    assert int(r['SourceRow']) == expected['source_row'] and int(r['SliceIndex']) == expected['slice']
    assert (r['OneDMatchMethod'],r['OneDAdSolver'],r['KrMode']) == ('ad','robust','dyn')
    assert float(r['HistoryMatchError']) >= 0
    files = {str(p):sha(p) for p in (root/'kr').rglob('*') if p.is_file()}
    write(root/'PASS.json',dict(passed=True,window=w,summary=r,files=files))


def combine_table(pc, kr):
    import numpy as np
    assert np.all(np.diff(pc[:,0])>0) and np.all(np.diff(kr[:,0])>0)
    assert pc[0,0] == kr[0,0] == 0 and abs(pc[-1,0]-kr[-1,0]) < 1e-12
    kr = kr.copy();kr[-1,0] = pc[-1,0]
    grid = np.unique(np.r_[pc[:,0],kr[:,0]])
    t = np.column_stack([grid,np.interp(grid,kr[:,0],kr[:,1]),np.interp(grid,kr[:,0],kr[:,2]),np.interp(grid,pc[:,0],pc[:,3])])
    pts = np.r_[grid,(grid[1:]+grid[:-1])/2]
    err = np.max(np.abs(np.interp(pts,pc[:,0],pc[:,3])-np.interp(pts,t[:,0],t[:,3])))
    assert err <= 1e-7+1e-12*np.max(pc[:,3])
    assert np.array_equal(t[np.searchsorted(grid,pc[:,0]),3],pc[:,3])
    assert t[0,1] == 0 and t[0,2] == 1 and t[-1,2] == 0
    assert np.all(np.diff(t[:,1])>=-1e-14) and np.all(np.diff(t[:,2])<=1e-14)
    assert np.all(np.diff(t[:,3])>=0) and np.isfinite(t).all()
    return t, float(err)


def assemble():
    import numpy as np
    from scipy.io import loadmat, savemat
    passes = [read(OUT/f'windows/w{w}/PASS.json') for w in range(1,7)]
    for p in passes:
        assert p['passed'];check_files(p['files'])
    B = loadmat(EXPORT/f'pc_tables_{CASE}.mat')
    tables = np.empty((6,87),dtype=object)
    reports=[]
    for w in range(1,7):
        path=OUT/f'windows/w{w}/kr/curves/kr_slice_curves_swi_medoid_{GEO}_cases_{REALIZATION:02}.csv'
        r=rows(path)
        assert len(r)==87*101
        for s in range(1,88):
            rs=[x for x in r if x['Window']==f'famp{w}' and int(x['SliceIndex'])==s]
            assert len(rs)==101
            kr=np.array([[float(x['GasSaturation']),float(x['Krg']),float(x['Krw'])] for x in rs])
            pc=B['pcTables'][w-1,s-1]
            # Kr columns are supplied exclusively by the fresh dynamic fits.
            pc4=np.column_stack([pc[:,0],np.zeros(len(pc)),np.zeros(len(pc)),pc[:,1]])
            t,err=combine_table(pc4,kr)
            tables[w-1,s-1]=t
            reports.append(dict(window=w,slice=s,points=len(t),max_pc_error_pa=err))
    out=OUT/'export';out.mkdir()
    B={k:v for k,v in B.items() if not k.startswith('__')}
    del B['pcTables']
    B['exportTables']=tables;B['krMethod']='production_dynamic_swi_medoid_recomputed'
    B['schema']='candidate_sgof_native_nodes_v1'
    file=out/f'candidate_tables_{CASE}.mat'
    savemat(file,B,do_compression=True)
    R=loadmat(file)['exportTables']
    assert all(np.array_equal(tables[w,s],R[w,s]) for w in range(6) for s in range(87))
    csvwrite(out/'curve_audit.csv',reports)
    csvwrite(out/'dynamic_fit_summary.csv',[p['summary'] for p in passes])
    write(out/'assembly_pass.json',dict(passed=True,cores=522,all_structural_checks_passed=True,production_fits=6,
        original_candidate_pc_sha256=sha(EXPORT/f'pc_tables_{CASE}.mat'),
        table_sha256=sha(file),max_pc_error_pa=max(x['max_pc_error_pa'] for x in reports),
        reservoir_submitted=False))
    _,cm=case_manifest()
    baseline=Path(cm['production']['specific_path'])
    matlab(OUT/'specific_process',f'make_specific({mq(baseline)},{mq(file)},'
           f'{mq(out/"assembly_pass.json")},{mq(out/f"candidate_specific_{CASE}.mat")});')
    receipt=read(Path(str(out/f'candidate_specific_{CASE}.mat')+'.roundtrip.json'))
    assert receipt['passed'] and receipt['production_input_unchanged']
    write(out/'specific_pass.json',receipt)


def qualify():
    assert read(OUT/'export/assembly_pass.json')['passed']
    assert read(OUT/'export/specific_pass.json')['passed']
    out=OUT/'export'
    common=tomllib.loads((PAIR/'campaign.toml').read_text())['common']['path']
    prefix=['bash','-lc']
    def julia(folder,script,args):
        cmd='module purge\nmodule load community-modules julia/1.10.4\n'
        cmd+='export JULIA_NUM_THREADS=2 JULIA_NUM_GC_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_CPU_TARGET=generic\n'
        cmd+=f'cd {shlex.quote(str(JUTUL))}\n'
        cmd+='exec '+shlex.join(['julia','--startup-file=no','--compiled-modules=no','--project=.',str(script)]+list(map(str,args)))
        command(folder,prefix+[cmd])
    julia(OUT/'runtime_process',SOURCE/'validate_constitutive.jl',
          [out/f'candidate_tables_{CASE}.mat',out/'runtime_validation',common,out/f'candidate_specific_{CASE}.mat'])
    julia(OUT/'smoke_process',EXPORT30/'smoke_validation_v3.jl',
          [out/f'candidate_tables_{CASE}.mat',out/'runtime_validation',out/'smoke_validation_v3','15,19,83,87,233,267,328'])
    q=tomllib.loads((out/'smoke_validation_v3/PASS.toml').read_text())
    assert q['passed'] and q['smoke_simulations']==14 and q['table_sha256']==sha(out/f'candidate_tables_{CASE}.mat')
    write(OUT/'qualified.json',dict(passed=True,table_sha256=q['table_sha256'],specific_sha256=sha(out/f'candidate_specific_{CASE}.mat')))


def toml_values(values):
    return ''.join(f'{k} = {str(v).lower() if isinstance(v,bool) else json.dumps(v)}\n' for k,v in values.items())


def launch():
    q=read(OUT/'qualified.json');assert q['passed']
    assert shutil.disk_usage(OUT).free>200*1024**3
    master,cm=case_manifest();production=cm['production']
    root=OUT/'reservoir';root.mkdir()
    for n in ('logs','tmp','source'):(root/n).mkdir()
    for p in (PAIR/'source').iterdir():
        if p.is_file():shutil.copy2(p,root/'source'/p.name)
    guard=(SOURCE/'candidate_guard.sh').read_text()
    assert all(guard.count(x)==1 for x in ('@ROOT@','@TASK@','@CASE@'))
    (root/'source/candidate_guard.sh').write_text(guard.replace('@ROOT@',str(root)).replace('@TASK@',str(TASK)).replace('@CASE@',CASE))
    for filename in ('submit_pair.sh','gom_step62_effective_pc_global_plateau_full.sbatch',
                     'gom_step62_effective_pc_global_plateau_preflight.sbatch'):
        p=root/'source'/filename;txt=p.read_text()
        assert txt.count('--array=1137')==(2 if filename=='submit_pair.sh' else 1)
        p.write_text(txt.replace('--array=1137',f'--array={TASK}'))
    full=root/'source/gom_step62_effective_pc_global_plateau_full.sbatch'
    txt=full.read_text();old='-name \'jutul_*.jld2\' | wc -l)" -eq 4'
    assert txt.count(old)==1
    full.write_text(txt.replace(old,'-name \'jutul_*.jld2\' | wc -l)" -eq 210'))
    original=tomllib.loads((PAIR/'campaign.toml').read_text());m=copy.deepcopy(original)
    oldid=m['campaign_id'];m['campaign_id']=f'hybrid_{CASE}_wrr{ANGLE}_20260911_v1'
    m['archive_root']=str(root);m['allowed_original_task']=TASK
    baseline=Path(m['baseline']['manifest'])
    assert sha(baseline)==m['baseline']['manifest_sha256']==master['sources'][str(baseline)]
    original_cases=tomllib.loads(baseline.read_text())['cases']
    expected=[c for c in original_cases if c['task']==TASK]
    assert len(expected)==1
    for key in ('case_key','geology_id','realization_id','geology_hash','specific_path','specific_sha256','level3_case_name'):
        assert expected[0][key]==production[key],key
    m['baseline']['specific_path']=production['specific_path']
    m['baseline']['specific_sha256']=production['specific_sha256']
    report=rows_tab(MOVIE_DATA/'report_times.tsv')
    assert [int(r['step']) for r in report]==list(range(1,211))
    m['workflow']['retain_years']=[float(r['time_years']) for r in report]
    spec=OUT/'export'/f'candidate_specific_{CASE}.mat';tables=OUT/'export'/f'candidate_tables_{CASE}.mat'
    assert sha(spec)==q['specific_sha256'] and sha(tables)==q['table_sha256']
    m['cases']=[copy.deepcopy(production)]
    m['cases'][0].update(specific_path=str(spec),specific_sha256=sha(spec),specific_bytes=spec.stat().st_size)
    text=toml_values({k:v for k,v in m.items() if not isinstance(v,(dict,list))})
    for k in ('baseline','workflow','common'):text+='\n['+k+']\n'+toml_values(m[k])
    text+='\n[[cases]]\n'+toml_values(m['cases'][0])
    manifest=root/'campaign.toml';manifest.write_text(text)
    Path(str(manifest)+'.sha256').write_text(sha(manifest)+'  campaign.toml\n')
    env={}
    for line in (PAIR/'locked.env').read_text().splitlines():
        if line.startswith('export '):
            k,v=line[7:].split('=',1);env[k]=shlex.split(v)[0] if v else ''
    env={k:v.replace(str(PAIR),str(root)).replace(oldid,m['campaign_id']) for k,v in env.items()}
    env.update(GOM_PRODUCTION_MANIFEST_SHA256=sha(manifest),
        GOM_PRODUCTION_TASK=str(TASK),GOM_PRODUCTION_CASE_KEY=CASE,
        GOM_PRODUCTION_GEOLOGY_ID=GEO,GOM_PRODUCTION_REALIZATION_ID=str(REALIZATION),
        GOM_PRODUCTION_GEOLOGY_HASH=production['geology_hash'],
        GOM_PRODUCTION_LEVEL3_CASE_NAME=production['level3_case_name'],
        BASELINE_SPECIFIC_MAT=production['specific_path'],
        GOM_PRODUCTION_SPECIFIC_MAT=str(spec),GOM_PRODUCTION_SPECIFIC_SHA256=sha(spec),
        GOM_PRODUCTION_SPECIFIC_BYTES=str(spec.stat().st_size),CANDIDATE_TABLES_MAT=str(tables),
        GOM_PRODUCTION_RETAIN_YEARS=','.join(map(repr,m['workflow']['retain_years'])))
    assert env['GOM_PRODUCTION_CASE_COUNT']=='1'
    assert env['GOM_PRODUCTION_ARCHIVE_ROOT']==str(root)
    assert env['GOM_PRODUCTION_MANIFEST_PATH']==str(manifest)
    assert not any('s03_c022' in v for k,v in env.items()), 'Intermediate identity leaked into new launcher'
    (root/'locked.env').write_text(''.join(f'export {k}={shlex.quote(str(v))}\n' for k,v in env.items()))
    protected={Path(m['common']['path']):m['common']['sha256'],
        Path(production['specific_path']):production['specific_sha256'],spec:sha(spec),tables:sha(tables),
        JUTUL/'Manifest.toml':m['jutul_manifest_sha256'],OUT/'qualified.json':sha(OUT/'qualified.json'),
        baseline:m['baseline']['manifest_sha256']}
    check_files(protected)
    (root/'INPUTS_SHA256.txt').write_text(''.join(f'{h}  {p}\n' for p,h in protected.items()))
    fs=[manifest,Path(str(manifest)+'.sha256'),root/'locked.env',root/'INPUTS_SHA256.txt']
    fs+=sorted(p for p in (root/'source').rglob('*') if p.is_file())
    (root/'STUDY_SHA256.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(root)}\n' for p in fs))
    check=tomllib.loads(manifest.read_text())
    assert {k:v for k,v in check['workflow'].items() if k!='retain_years'}=={k:v for k,v in original['workflow'].items() if k!='retain_years'}
    assert check['common']==original['common']==master['common'] and check['restart_reuse'] is False
    write(root/'comparison_contract.json',dict(passed=True,case=CASE,angle=ANGLE,
        baseline_specific_sha256=production['specific_sha256'],baseline_task=TASK,
        same_draws=True,pc_convention='physical_prebreakthrough',host_rock_unchanged=True,
        kr_method='fresh production dynamic six-medoid fitting and 522 endpoint mapping',
        candidate_pc_preserved=True,movie_report_states=210,cpus=8,memory_gb=18))
    env2=dict(os.environ,STUDY_ROOT=str(root),GOM_GRID_ROOT=str(root),JUTULDARCY_COMBINED_REPO=str(JUTUL))
    command(OUT/'reservoir_submission',['bash',str(root/'source/submit_pair.sh')],env2)


def rows_tab(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f,delimiter='\t'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['seal','prepare','window','assemble','qualify','launch']);p.add_argument('window',nargs='?',type=int,default=0)
    a=p.parse_args()
    if a.mode=='seal':seal();sys.exit(0)
    verify()
    if a.mode=='window':window(a.window)
    else:globals()[a.mode]()
    verify()
