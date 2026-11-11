import common as C
base=C.CODE.parent;invest=base/'phase1_remaining56_investigation_20261111_v1'
pilot=base/'phase1_precision_recovery_20261111_v1'
s=C.read(invest/'SNAPSHOT.json');classification=C.read(invest/'CLASSIFICATION.json')
excluded=set(C.read(base/'phase1_direct_saved_20261111_v1/SCOPE.json')['tests'][i]['key'] for i in range(3))
excluded.add('s02_c003_pc04_ca40_IND_005_core229')
keys=sorted(r['key'] for r in classification['cores'] if r['stage']=='primary' and r['key'] not in excluded)
C.require(len(keys)==8,'Fixed eight remaining primary failures')
rows={}
for key in keys:
    r=next(x for x in s['held'] if x['key']==key)
    C.require(not r['probe']['PROBE_RESULT.json']['passed'] and r['probe']['PROBE_RESULT.json']['failed_stage']=='primary','Wrong primary failure')
    rows[key]=dict(category='remaining_primary_failure',prior_scientific_failures=2,
        metadata={p:h for p,h in s['metadata_hashes'].items() if key in p})
C.require(not (C.CODE/'SCOPE.json').exists(),'Already frozen')
C.write(C.CODE/'SCOPE.json',dict(keys=keys,new_keys=keys,rows=rows,batches=[keys],
    pilot_closeout_sha256=C.sha(pilot/'results/output/CLOSEOUT.json')))
C.write(C.CODE/'AUTHORIZATION.json',dict(approved_by_user=True,keys=keys,publication=True,exclude_core229=True,
    request='Please move forward carefully and do not stop until you complete all the cases.',
    scope='Eight remaining primary failures after three direct full recoveries publish; one bounded third attempt each, preserve all completed calculations'))
