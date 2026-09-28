"""Only release attempts proven to have stopped before scientific calculation."""
from launch_support import *

guard()
with single_writer():
    db=connection();active={r[0] for r in queue()};result=[]
    for name,kind in db.execute('select id,kind from batches').fetchall():
        batch=ROOT/'batches'/name;spec=read(batch/'spec.json')
        jobs=[r[0] for r in db.execute('select job from jobs where batch=?',(name,))]
        if any(j in active for j in jobs):continue
        if kind=='fresh':
            p=batch/'prepare.log'
            if not p.exists() or 'Precompiled image' not in p.read_text() or 'IterativeSolvers' not in p.read_text():continue
            assert not (batch/'planned').exists() and not (batch/'curves').exists()
        else:
            p=batch/'export.log'
            if not p.exists():
                if jobs!=['22867970']:continue
                text=subprocess.check_output(['sacct','-nP','-X','-j',jobs[0],'--format=State'],text=True)
                assert text.strip().startswith('CANCELLED')
                p=ROOT/'submissions'/(name+'_export.submitted.json')
            elif "No module named 'tomli'" not in p.read_text():continue
            assert not (batch/'export').exists()
        assert not list((batch/'receipts').glob('*.json'))
        for item in spec['items']:
            row=db.execute('select status,batch from owners where key=?',(item['key'],)).fetchone()
            assert row==('CLAIMED',name)
            assert not (ROOT/'accepted'/(item['key']+'.json')).exists()
            db.execute("update owners set status='READY',batch=NULL where key=?",(item['key'],))
        db.execute("update batches set status='SUPERSEDED_LAUNCH_ONLY' where id=?",(name,))
        result.append(dict(batch=name,kind=kind,cores=len(spec['items']),evidence=str(p),sha256=sha(p)))
    db.commit();db.close()
    write(ROOT/'audit/LAUNCH_ONLY_RECOVERY.json',dict(passed=True,batches=result,science_rerun=False,old_outputs_preserved=True))
    event('LAUNCH_ONLY_RECOVERY',batches=result)
