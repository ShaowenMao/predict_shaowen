"""One-time, read-only selection from saved classification; no new case draws."""
from collections import Counter
from pathlib import Path
import common as C

base=C.CODE.parent
invest=base/'phase1_remaining56_investigation_20261111_v1'
pilot=base/'phase1_precision_recovery_20261111_v1'
classification=C.read(invest/'CLASSIFICATION.json');snapshot=C.read(invest/'SNAPSHOT.json')
pilot_keys=C.read(pilot/'SCOPE.json')['keys']
rows=[r for r in classification['cores'] if r['investigation_category']=='stable_reference_plateau']
counts=Counter(r['case_id'] for r in classification['cores'])
new=[r['key'] for r in sorted(rows,key=lambda r:(counts[r['case_id']],r['case_id'],r['key'])) if r['key'] not in pilot_keys]
keys=pilot_keys+new
C.require(len(rows)==38 and len(new)==35 and len(set(keys))==38,'Fixed plateau classification differs')
C.require('s02_c003_pc04_ca40_IND_005_core229' not in keys,'Core229 must stay isolated')
details={}
for key in keys:
    saved=next(r for r in snapshot['held'] if r['key']==key)
    d=saved['probe']['PROBE_RESULT.json']
    C.require(not d['passed'] and d['failed_stage']=='reference','Wrong failure scope')
    details[key]=dict(category='stable_reference_plateau',prior_scientific_failures=2,
        operator_sha256=saved['probe']['CAPTURE.json']['operator_sha256'],
        metadata={p:h for p,h in snapshot['metadata_hashes'].items() if key in p})
C.require(not (C.CODE/'SCOPE.json').exists(),'Scope already frozen')
C.write(C.CODE/'SCOPE.json',dict(keys=keys,new_keys=new,rows=details,
    classification_sha256=C.sha(invest/'CLASSIFICATION.json'),snapshot_sha256=C.sha(invest/'SNAPSHOT.json'),
    pilot_closeout_sha256=C.sha(pilot/'results/output/CLOSEOUT.json'),batches=[new[i:i+8] for i in range(0,35,8)]))
C.write(C.CODE/'AUTHORIZATION.json',dict(approved_by_user=True,keys=keys,publication=True,exclude_core229=True,
    request='Publish the three verified results, then recover other cores with the same precision issue in targeted batches; do not recalculate completed results.',
    scope='3 qualified archives plus 35 stable reference plateaus; all other holds excluded'))
print('Fixed scope: 3 publish-only, 35 new recoveries in 8/8/8/8/3 batches')
