"""Single-writer, content-addressed campaign dispatch with fail-closed class ramp."""
import time
from launch_support import *

def policy(kind,cl):
    if kind=='derive':return {'export':dict(cpus=1,memory_mib=512,minutes=240,deps=[]),
        'native':dict(cpus=1,memory_mib=3072,minutes=120,deps=['export'])}
    flow,prepare=CLASSES[cl]
    return {'prepare':dict(cpus=2,memory_mib=prepare,minutes=120,deps=[]),
        'flow':dict(cpus=2,memory_mib=flow,minutes=1440,deps=['prepare']),
        'export':dict(cpus=1,memory_mib=512,minutes=120,deps=['flow']),
        'native':dict(cpus=1,memory_mib=3072,minutes=120,deps=['export'])}

def stage_order(kind):return ['export','native'] if kind=='derive' else ['prepare','flow','export','native']

def returned(name):
    p=ROOT/'submissions'/(name+'.returned.json')
    if p.exists():return read(p)['job']
    intent=ROOT/'submissions'/(name+'.intent.json')
    if not intent.exists():return None
    t=read(intent)['token']
    rows=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%k'],text=True,timeout=30)
    matches=[r.split('|')[0] for r in rows.splitlines() if r.split('|')[-1]==t]
    assert len(matches)==1,'Uncertain submission held, never resubmit automatically: '+name
    job=matches[0];assert job.isdigit()
    write(p,dict(job=job,token=t,recovered=True));return job

def submit(name,batch,stage,pol,own,rows=None):
    job=returned(name)
    if job:return job
    a=admission(queue() if rows is None else rows,own,pol['cpus'],pol['memory_mib'])
    if not a['passed']:return None
    token='pc100kr10:'+sha(HERE/'LAUNCH_LOCK.json')[:12]+':'+name
    mode='controller' if stage=='controller' else 'index' if stage=='index' else 'stage'
    cmd=['sbatch','--parsable','--partition=mit_normal','--account='+ACCOUNT,'--qos='+ACCOUNT,
         '--nodes=1','--ntasks=1','--cpus-per-task='+str(pol['cpus']),
         '--mem='+str(pol['memory_mib'])+'M','--time='+str(pol['minutes']),
         '--exclude=node1622','--no-requeue','--comment='+token,'--job-name=k10_'+name,
         '--output='+str(ROOT/'logs'/(name+'_%j.out')),str(HERE/'campaign_job.sh'),str(HERE),mode,batch,stage]
    write(ROOT/'submissions'/(name+'.intent.json'),dict(command=cmd,token=token,admission=a))
    r=subprocess.run(cmd,text=True,capture_output=True,check=True,timeout=60)
    job=r.stdout.strip().split(';')[0];assert job.isdigit(),r.stdout
    write(ROOT/'submissions'/(name+'.returned.json'),dict(job=job,token=token,stdout=r.stdout))
    actual=subprocess.check_output(['scontrol','show','job','-o',job],text=True,timeout=30)
    resource().parse_allocation(actual,pol['cpus'],pol['memory_mib'])
    write(ROOT/'submissions'/(name+'.submitted.json'),dict(job=job,actual_scontrol=actual,policy=pol))
    event('SUBMITTED',stage_name=name,job=job,cpus=pol['cpus'],memory_mib=pol['memory_mib']);return job

def make_batch(db,kind,cl):
    limit=32 if kind=='derive' else 4
    q='select key,item_json from owners where status=? and kind=?'
    args=['READY',kind]
    if kind=='fresh':q+=' and class=?';args.append(cl)
    q+=' order by key limit ?';args.append(limit)
    rows=db.execute(q,args).fetchall()
    if not rows:return None
    name='b'+str(db.execute('select count(*) from batches').fetchone()[0]+1).zfill(6)
    p=ROOT/'batches'/name/'spec.json'
    spec=dict(id=name,kind=kind,class_name=cl,items=[json.loads(r[1]) for r in rows],
              stages=policy(kind,cl),flow_mib=CLASSES[cl][0] if kind=='fresh' else 0,
              launch_lock_sha256=sha(HERE/'LAUNCH_LOCK.json'))
    # Immutable spec precedes the transaction. Re-entry verifies it instead of replacing it.
    write(p,spec)
    db.execute('insert into batches values(?,?,?,?,?,?)',(name,cl,kind,str(p),'ACTIVE',time.time()))
    for key,_ in rows:
        cur=db.execute('update owners set status=?,batch=? where key=? and status=?',('CLAIMED',name,key,'READY'))
        assert cur.rowcount==1
    db.commit();event('BATCH_CLAIMED',batch=name,kind=kind,resource_class=cl,cores=len(rows));return name

def publish(db,name):
    if db.execute('select 1 from imported_batches where id=?',(name,)).fetchone():return
    stage_receipt(name,'native');spec=read(ROOT/'batches'/name/'spec.json')
    export=read(ROOT/'batches'/name/'export/EXPORT_BATCH.json')
    passed={r['key']:r for r in export['records']}
    for item in spec['items']:
        key=item['key']
        if key not in passed:
            db.execute('update owners set status=? where key=?',('HOLD_STAGE_REVIEW',key));continue
        x=passed[key];pins_ok(x['artifacts'])
        p=ROOT/'accepted'/(key+'.json')
        receipt=dict(passed=True,key=key,old_numerical_key=item['old_numerical_key'],batch=name,
            method_sha256=method_hash(),export_receipt=str(ROOT/'batches'/name/'export'/key/'EXPORT_PASS.json'),
            export_sha256=sha(ROOT/'batches'/name/'export'/key/'EXPORT_PASS.json'),
            native_receipt=str(receipt_path(name,'native')),native_sha256=sha(receipt_path(name,'native')),
            artifacts=x['artifacts'],source_kind=spec['kind'],flow_recomputed=spec['kind']=='fresh',
            exact_aliases=[r['key'] for r in [item['row'],*item['aliases']]])
        write(p,receipt)
        db.execute('update owners set status=? where key=?',('ACCEPTED',key))
        db.execute("update placement_status set status='KR10_ACCEPTED',reason='Exact input identity and native import verified' where numerical_key=?",(item['old_numerical_key'],))
    cl=spec['class_name'] if spec['kind']=='fresh' else 'derive'
    db.execute('insert into imported_batches values(?,?,?,?)',(name,cl,str(receipt_path(name,'native')),sha(receipt_path(name,'native'))))
    db.execute('update batches set status=? where id=?',('PASS' if len(passed)==len(spec['items']) else 'PARTIAL_HOLD',name))
    db.commit();event('BATCH_ACCEPTED',batch=name,accepted=len(passed),total=len(spec['items']))

def tick():
    guard()
    if not (ROOT/'audit/INDEX_PASS.json').exists():return False
    assert read(ROOT/'audit/INDEX_PASS.json')['passed']
    with single_writer():
        db=connection();rows=queue();active={r[0] for r in rows}
        own={r[0] for r in db.execute('select job from jobs')}
        own.update(read(p)['job'] for p in (ROOT/'submissions').glob('*.returned.json'))
        if not receipt_path('bootstrap','bootstrap').exists():
            job=returned('bootstrap')
            if job and job not in active:
                event('HOLD_BOOTSTRAP',job=job);db.close();return False
            if not job:submit('bootstrap','bootstrap','bootstrap',dict(cpus=2,memory_mib=2816,minutes=30),own)
            db.close();return False
        stage_receipt('bootstrap','bootstrap')
        if not receipt_path('cache','cache').exists():
            job=returned('cache')
            if job and job not in active:event('HOLD_CACHE',job=job)
            if not job:submit('cache','cache','cache',dict(cpus=2,memory_mib=2816,minutes=60),own)
            db.close();return False
        stage_receipt('cache','cache')
        for name, in db.execute("select id from batches where status='ACTIVE'").fetchall():
            spec=read(ROOT/'batches'/name/'spec.json')
            if receipt_path(name,'native').exists():publish(db,name);continue
            for stage in stage_order(spec['kind']):
                pol=spec['stages'][stage]
                token=name+'_'+stage
                if receipt_path(name,stage).exists():continue
                job=returned(token)
                if job:
                    if job not in active:
                        # Scheduler absence is not success; preserve output and isolate this batch.
                        db.execute('update batches set status=? where id=?',('HOLD_MISSING_RECEIPT',name));db.commit()
                        event('HOLD_MISSING_RECEIPT',batch=name,stage=stage,job=job)
                    break
                if any(not receipt_path(name,d).exists() for d in pol['deps']):break
                for d in pol['deps']:stage_receipt(name,d)
                job=submit(token,name,stage,pol,own)
                if job:
                    db.execute('insert or ignore into jobs values(?,?,?,?,?)',(token,name,job,'SUBMITTED',json.dumps(pol)));db.commit();own.add(job)
                break
        # Successful real campaign batches unlock more work in that exact resource class.
        for kind,cl in [('derive','derive'),*[('fresh',c) for c in CLASSES]]:
            passed=db.execute("select count(*) from batches where status='PASS' and kind=? and class=?",(kind,cl)).fetchone()[0]
            live=db.execute("select count(*) from batches where status='ACTIVE' and kind=? and class=?",(kind,cl)).fetchone()[0]
            held=db.execute("select count(*) from batches where status like '%HOLD%' and kind=? and class=?",(kind,cl)).fetchone()[0]
            allowed=concurrency(passed)
            if held:allowed=0
            # Admit before claiming. This prevents flooding the queue with resource-ineligible work.
            start=policy(kind,cl)['export' if kind=='derive' else 'prepare']
            created=0
            while live<allowed and created<16:
                if not admission(queue(),own,start['cpus'],start['memory_mib'])['passed']:break
                name=make_batch(db,kind,cl)
                if name is None:break
                stage='export' if kind=='derive' else 'prepare'
                job=submit(name+'_'+stage,name,stage,start,own)
                if job:
                    db.execute('insert into jobs values(?,?,?,?,?)',(name+'_'+stage,name,job,'SUBMITTED',json.dumps(start)));db.commit();own.add(job)
                live+=1;created+=1
        states=db.execute('select kind,status,count(*) from owners group by kind,status').fetchall()
        classes=db.execute('select class,status,count(*) from batches group by class,status').fetchall()
        placements=dict(db.execute('select status,count(*) from placement_status group by status').fetchall())
        count=db.execute("select count(*) from owners where status in ('READY','CLAIMED')").fetchone()[0]
        write(ROOT/'STATUS.json',dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             owners=states,classes=classes,placements=placements,admission=admission(queue(),own,0,0),
             full_design_cases=2430,full_design_placements=1268460,
             scope='All reconciled owner calculations; unprepared or unsupported placements explicitly held',
             exact_reuse_not_assumed_for_unprepared_inputs=True),replace=True)
        db.close();return count==0

if __name__=='__main__':
    guard()
    if len(sys.argv)>1 and sys.argv[1]=='launch':
        if not (ROOT/'audit/INDEX_PASS.json').exists():
            submit('index','index','index',dict(cpus=1,memory_mib=2048,minutes=120),set())
        controllers=list((ROOT/'submissions').glob('controller*.returned.json'))
        active={r[0] for r in queue()}
        if not any(read(p)['job'] in active for p in controllers):
            submit('controller'+str(len(controllers)).zfill(2),'controller','controller',dict(cpus=1,memory_mib=1024,minutes=1440),set())
    else:
        deadline=time.monotonic()+(23*3600 if len(sys.argv)>1 and sys.argv[1]=='watch' else 0)
        while True:
            if tick():break
            if time.monotonic()>=deadline:
                if len(sys.argv)>1 and sys.argv[1]=='watch':
                    existing=list((ROOT/'submissions').glob('controller*.returned.json'))
                    name='controller'+str(len(existing)).zfill(2)
                    submit(name,'controller','controller',dict(cpus=1,memory_mib=1024,minutes=1440),
                           {read(p)['job'] for p in (ROOT/'submissions').glob('*.returned.json')})
                break
            time.sleep(30)
