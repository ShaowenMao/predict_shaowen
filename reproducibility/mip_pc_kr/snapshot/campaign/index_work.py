"""Index successful old flow work and explicitly hold any ambiguous partial attempt."""
import re
from launch_support import *

def main():
    guard();allocation(1,2048)
    out=ROOT/'state';out.mkdir(exist_ok=True)
    p=out/'dispatch.sqlite';assert not p.exists()
    db=sqlite3.connect(p);db.execute('PRAGMA cache_size=-16384')
    src=sqlite3.connect('file:'+str(ROOT/'audit/reconciliation.sqlite')+'?mode=ro&immutable=1',uri=True)
    report=read(ROOT/'audit/RECONCILED.json');assert report['passed']
    assert sha(ROOT/'audit/reconciliation.sqlite')==report['database_sha256']
    src.backup(db);src.close()
    db.executescript('''
      CREATE TABLE owners(key TEXT PRIMARY KEY, old_key TEXT UNIQUE, class TEXT, kind TEXT, status TEXT, batch TEXT, item_json TEXT);
      CREATE TABLE batches(id TEXT PRIMARY KEY, class TEXT, kind TEXT, spec TEXT, status TEXT, created REAL);
      CREATE TABLE jobs(stage TEXT PRIMARY KEY, batch TEXT, job TEXT UNIQUE, status TEXT, policy_json TEXT);
      CREATE TABLE seen_attempts(key TEXT PRIMARY KEY, evidence TEXT);
      CREATE TABLE legacy_flows(old_key TEXT PRIMARY KEY, receipt TEXT, sha256 TEXT);
      CREATE TABLE imported_batches(id TEXT PRIMARY KEY, class TEXT, receipt TEXT, sha256 TEXT);
    ''')
    roots=[OLD/'numerical_wave01_reuse_v1/staged_dispatch_v1',OLD/'expansion_20260915_v2']
    paths=list((roots[0]/'flows').glob('*.json'))
    for root in roots:
        paths.extend(root.glob('batches/*/results/flows/*.json'))
        paths.extend(root.glob('warm_dispatch_wave01_20260915_v1/batches/*/flows/*.json'))
        paths.extend(root.glob('progressive_warm_wave02_20260915_v1/batches/*/results/flows/*.json'))
    problems=[]
    for i,p in enumerate(paths):
        try:
            r=read(p)
            assert r['complete'] and r['report']['passed'] and r['memory']['passed']
            key=r['numerical_key'];assert db.execute('select 1 from records where numerical_key=?',(key,)).fetchone()
            db.execute('insert or ignore into legacy_flows values(?,?,?)',(key,str(p),sha(p)))
        except (KeyError,AssertionError,FileNotFoundError) as e:problems.append(dict(path=str(p),reason=repr(e)))
        if i%200==0:event('INDEX_FLOW',count=i);db.commit()
    # A prior attempted core without a passing receipt cannot be resubmitted blindly.
    for root in roots:
        for p in (root/'attempts').glob('*'):
            m=re.search(r'(s\d+_c\d+_case\d+_core\d+)',p.name)
            if m:db.execute('insert or ignore into seen_attempts values(?,?)',(m[1],str(p)))
        for pattern in ['batches/*/cores/*','warm_dispatch_wave01_20260915_v1/batches/*/cores/*',
                        'progressive_warm_wave02_20260915_v1/batches/*/cores/*']:
            for p in root.glob(pattern):
                if p.is_dir() and re.fullmatch(r's\d+_c\d+_case\d+_core\d+',p.name):
                    db.execute('insert or ignore into seen_attempts values(?,?)',(p.name,str(p)))
    count=0
    for (old,) in db.execute('select distinct numerical_key from records').fetchall():
        rows=[json.loads(r[0]) for r in db.execute('select row_json from records where numerical_key=? order by key',(old,))]
        a=db.execute('select receipt,receipt_sha256 from accepted where numerical_key=?',(old,)).fetchone()
        if a is None:a=db.execute('select receipt,sha256 from legacy_flows where old_key=?',(old,)).fetchone()
        row=rows[0];cl,reason=route(row);kind='derive' if a else 'fresh';status='READY'
        attempted=[r['key'] for r in rows if db.execute('select 1 from seen_attempts where key=?',(r['key'],)).fetchone()]
        if not a and attempted:status='HOLD_LEGACY_PARTIAL'
        elif not a and reason:status=reason
        key=method_key(old)
        item=dict(key=key,old_numerical_key=old,row=row,aliases=rows[1:],kind=kind,class_name=cl,attempted=attempted)
        if a:item.update(receipt=a[0],receipt_sha256=a[1])
        db.execute('insert into owners values(?,?,?,?,?,?,?)',(key,old,cl,kind,status,None,json.dumps(item)))
        count+=1
        if count%500==0:db.commit();event('INDEX_OWNER',count=count)
    db.commit()
    states=db.execute('select kind,status,class,count(*) from owners group by kind,status,class').fetchall()
    db.close()
    r=dict(passed=True,owners=count,states=states,unaccepted_complete_flow_receipts=len(paths),
        legacy_receipt_review=problems,memory=memory(2048),source_database_sha256=report['database_sha256'],
        unmapped_placements_explicitly_held_pending_replay=True,old_results_modified=False)
    write(ROOT/'audit/INDEX_PASS.json',r);event('INDEX_PASS',owners=count,states=states)

if __name__=='__main__':main()
