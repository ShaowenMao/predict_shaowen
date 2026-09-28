"""Bounded-memory inventory reconciliation. Never declare a digest-only cache hit."""
import collections
import sqlite3
from common_launch import *

def resolve_receipt(p):
    x=read(p)
    if 'receipt' in x and 'sha256' in x:
        assert sha(x['receipt'])==x['sha256'];p=Path(x['receipt']);x=read(p)
    return p,x

def main():
    verify();actual=allocation(1,2048)
    spec=read(REPLAY/'spec.json');assert sha(spec['inventory'])==spec['inventory_sha256']
    manifest=read(REPLAY/'campaign.json')['cases'];assert len(manifest)==2430
    out=ROOT/'audit';out.mkdir(parents=True,exist_ok=True)
    path=out/'reconciliation.sqlite';assert not path.exists()
    db=sqlite3.connect(path);db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA cache_size=-32768')
    source=sqlite3.connect('file:'+spec['inventory']+'?mode=ro&immutable=1',uri=True)
    source.backup(db);source.close()
    assert db.execute('SELECT count(*) FROM assignments').fetchone()[0]==1268460
    assert db.execute('SELECT count(*) FROM cases').fetchone()[0]==2430
    for c in manifest:
        assert db.execute('SELECT case_key,specific_sha256 FROM cases WHERE task=?',(c['task'],)).fetchone()==(c['case_key'],c['specific_sha256'])
        assert db.execute('SELECT count(*) FROM assignments WHERE task=?',(c['task'],)).fetchone()[0]==522
    db.executescript('''
      CREATE TABLE records(key TEXT PRIMARY KEY, case_key TEXT, slot INTEGER, numerical_key TEXT, resource_class TEXT, row_json TEXT);
      CREATE INDEX records_num ON records(numerical_key);
      CREATE TABLE accepted(numerical_key TEXT PRIMARY KEY, receipt TEXT, receipt_sha256 TEXT, data_json TEXT);
      CREATE TABLE sources(path TEXT PRIMARY KEY, sha256 TEXT);
      CREATE TABLE placement_status(task INTEGER, slot INTEGER, case_key TEXT, replay_group TEXT, numerical_key TEXT, status TEXT, reason TEXT, PRIMARY KEY(task,slot));
      CREATE INDEX placement_num ON placement_status(numerical_key);
    ''')
    db.execute("INSERT INTO placement_status SELECT task,(window-1)*87+slice,case_key,candidate_replay_key,NULL,'WAIT_REPLAY','Exact binary identity not yet available' FROM assignments")
    bycase={c['case_key']:c for c in manifest}
    def add_source(p):db.execute('INSERT OR REPLACE INTO sources VALUES (?,?)',(str(p),sha(p)))
    def add_record(row):
        key=row['key'];case=row['case'];slot=row['slot'];task=bycase[case]['task']
        x=db.execute('SELECT sample,seed FROM assignments WHERE task=? AND (window-1)*87+slice=?',(task,slot)).fetchone()
        assert x==(row['sample'],row['seed'])
        p=Path(row['replay_receipt']);assert sha(p)==row['replay_receipt_sha256']
        r=read(p);assert r['complete'] and r['memory']['passed'] and r['case']==case and r['slot']==slot
        for name,col in [('prepared/raw_unplanned.bin','raw'),('prepared/geometry_unplanned.bin','geometry')]:
            assert str(Path(r['output'])/name)==row[col] and r['files'][name]==row[col+'_hash']
        old=db.execute('SELECT numerical_key FROM records WHERE key=?',(key,)).fetchone()
        assert old is None or old[0]==row['numerical_key']
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)',(key,case,slot,row['numerical_key'],row['resource']['stratum'],json.dumps(row)))
        db.execute("UPDATE placement_status SET numerical_key=?,status='PREPARED_PENDING_VERIFY',reason='Recheck input bytes before numerical dispatch' WHERE task=? AND slot=?",(row['numerical_key'],task,slot))
    p=OLD/'numerical_wave01_reuse_v1/PLAN.json';old=read(p);add_source(p)
    for row in old['placements'].values():add_record(row)
    event('OLD_PLACEMENTS_INDEXED',count=db.execute('SELECT count(*) FROM records').fetchone()[0]);db.commit()
    for i,p in enumerate(sorted((OLD/'expansion_20260915_v2/registered').glob('*.json'))):
        x=read(p);assert x['passed'];add_source(p)
        for row in x['placements']:add_record(row)
        if i%50==0:db.commit();event('REGISTERED_BATCHES_INDEXED',count=i)
    roots=[OLD/'numerical_wave01_reuse_v1',OLD/'numerical_wave01_reuse_v1/staged_dispatch_v1',OLD/'expansion_20260915_v2']
    candidates=[]
    for root in roots:candidates.extend((root/'accepted').glob('*.json'))
    staged=roots[1]
    candidates.extend((staged/'warm_batch_integration_20260915_v1/accepted').glob('*.json'))
    candidates.extend((staged/'warm_dispatch_wave01_20260915_v1/batches').glob('*/accepted/*.json'))
    candidates.extend((staged/'progressive_warm_wave02_20260915_v1/accepted_index').glob('*.json'))
    candidates.extend((staged/'progressive_warm_wave02_20260915_v1/batches').glob('*/results/accepted/*.json'))
    rejected=[]
    for p in candidates:
        try:
            rp,x=resolve_receipt(p);assert x['complete'] and x['native_passed'] and x['memory']['passed']
            key=x['numerical_key'];assert db.execute('SELECT 1 FROM records WHERE numerical_key=?',(key,)).fetchone()
            db.execute('INSERT OR IGNORE INTO accepted VALUES (?,?,?,?)',(key,str(rp),sha(rp),json.dumps(x)))
        except (KeyError,AssertionError,FileNotFoundError) as e:rejected.append(dict(path=str(p),reason=repr(e)))
    db.execute("UPDATE placement_status SET status='RAW100_REUSE_PENDING_VERIFY',reason='Accepted legacy result: migrate only after bound artifact checks' WHERE numerical_key IN (SELECT numerical_key FROM accepted)")
    # Unprepared benchmark families stay explicit, not silently omitted from 2430.
    for c in manifest:
        if c['realization_id'] in (101,102):
            db.execute("UPDATE placement_status SET status='HOLD_CASE_ADAPTER',reason='Legacy generic replay adapter not qualified for this benchmark family' WHERE task=? AND numerical_key IS NULL",(c['task'],))
    db.commit()
    statuses=dict(db.execute('SELECT status,count(*) FROM placement_status GROUP BY status').fetchall())
    count=db.execute('SELECT count(DISTINCT numerical_key) FROM records').fetchone()[0]
    refs={str(p):sha(p) for p in (REPLAY/'source_lock.json',NUM/'source_lock.json',QUAL/'COMPLETE_REVIEW.json')}
    db.close()
    result=dict(passed=True,cases=2430,placements=1268460,known_binary_owner_keys=count,placement_status=statuses,
                receipt_conflicts_or_missing=rejected,source_pins=refs,database_sha256=sha(path),memory=memory(2048),
                actual_scontrol=actual,old_campaign_modified=False,production_modified=False,
                unprepared_binary_identity_unknown=True,all_accepted_results_require_artifact_verification_before_reuse=True)
    write(out/'RECONCILED.json',result);event('RECONCILIATION_PASS',**{k:v for k,v in result.items() if k not in ('actual_scontrol','source_pins','receipt_conflicts_or_missing')})

if __name__=='__main__':main()
