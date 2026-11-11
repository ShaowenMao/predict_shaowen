import tomllib
import common as C
from primary_proof import passing_primary
base=C.CODE.parent
reference=C.read(base/'phase1_precision_reference5_20261111_v1/SCOPE.json')
iterations=C.read(base/'phase1_remaining56_tests_20261111_v1/SCOPE.json')
keys=['s02_c003_pc04_ca40_IND_002_core096','s02_c003_pc04_ca40_IND_005_core229',
      's02_c010_pc04_ca40_IND_005_core192','s02_c010_pc04_ca40_IND_008_core221']
rows={}
for key in keys:
    slot=key[-3:];it=slot=='229'
    parent='phase1_remaining56_tests_20261111_v1' if it else 'phase1_precision_reference5_20261111_v1'
    root='/orcd/data/juanes/001/shaowen/'+parent+'/output/'+key+('' if it else '/work/recovery/precision')
    fixture=C.CODE/'fixtures'/(slot+'.toml');d=tomllib.loads(fixture.read_text())
    stage='primary_1.0e-14' if slot=='096' else 'primary_1.0e-12'
    prior=next(x for x in iterations['tests'] if x['key']==key) if it else reference['rows'][key]
    row=dict(key=key,category='saved_primary_failed_reference',prior_scientific_failures=3,
        saved_mode='iterations' if it else 'precision',saved_source=C.sha(base/parent/'LOCK.json'),
        saved_result=root+'/RESULT.toml',saved_primary=root+'/'+stage+'_pressure.jls',primary_stage=stage,
        primary_shift=1e-14 if slot=='096' else 1e-12,parts_hash=d['parts_hash'],operator_sha256=d['operator_sha256'],
        metadata=dict(prior['metadata']))
    row['metadata'][row['saved_result']]=C.sha(fixture)
    row['metadata'][row['saved_primary']]=C.sha(C.CODE/'fixtures'/(slot+'_pressure.jls'))
    row['primary_q']=passing_primary(d,row)
    rows[key]=row
C.require(not (C.CODE/'SCOPE.json').exists(),'Already frozen')
C.write(C.CODE/'SCOPE.json',dict(keys=keys,new_keys=keys,rows=rows,batches=[keys],
    previous_jobs=['28665837','28666059','28666060','28650395'],
    pilot_closeout_sha256=reference['pilot_closeout_sha256']))
C.write(C.CODE/'AUTHORIZATION.json',dict(approved_by_user=True,keys=keys,publication=True,allow_core229=True,
    reference_attempts_per_core=1,prior_failures_per_core=3,new_bounded_attempt=4,
    request='Approved. Please run one isolated direct-reference test for each of the four remaining cores. Reuse saved primary solutions, keep scientific checks unchanged, and finish the cases if they pass without recalculating completed results.'))
