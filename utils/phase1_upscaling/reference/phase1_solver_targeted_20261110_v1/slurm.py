"""Held submissions: one immutable intent, one ACK, no blind resubmission."""
import datetime
import getpass
import re
import subprocess
import common as C

def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=90).strip()

def spec(label,mode,args,array=None,dependency=None,cpus=1,memory=1024,minutes=10):
    C.require(re.fullmatch(r'[A-Za-z0-9_]+',label) is not None,'Unsafe label')
    return dict(label=label,name='p1q_'+label,mode=mode,args=args,array=array,dependency=dependency,cpus=cpus,
        memory=memory,minutes=minutes,root=str(C.ROOT),source=C.sha(C.CODE/'LOCK.json'),script=str(C.CODE/'job.sh'))

def check(fields,want):
    for k,v in dict(Account='mit_amf_advanced_cpu',QOS='mit_amf_advanced_cpu',Partition='mit_normal',
                    NumCPUs=str(want['cpus']),Requeue='0',JobName=want['name'],Command=want['script']).items():
        C.require(fields.get(k)==v,'Submitted allocation mismatch: '+k)
    m=re.fullmatch(r'(\d+)(M|G)',fields.get('MinMemoryNode',''))
    C.require(m and int(m[1])*(1024 if m[2]=='G' else 1)==want['memory'],'Submitted memory mismatch')
    C.require(fields.get('ArrayTaskThrottle','0')=='0','Unexpected array throttle')
    if want['array'] is None:C.require('ArrayJobId' not in fields,'Unexpected array')
    else:C.require(fields.get('ArrayTaskId')==want['array'],'Array mismatch')
    minutes=want['minutes'];wall=f'{minutes//60:02d}:{minutes%60:02d}:00'
    C.require(fields['TimeLimit']==wall,'Wall time mismatch')
    if want['dependency']:
        dep=fields.get('Dependency','')
        if dep=='(null)':
            states=command(['sacct','-X','-n','-P','-j',want['dependency'],'--format=State'])
            C.require(states and all(x.strip('| ').split()[0] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED') for x in states.splitlines()),'Dependency vanished before terminality')
        else:C.require(dep.startswith('afterany:'+want['dependency']) and ',' not in dep,'Unexpected dependency')
    else:C.require(fields.get('Dependency')=='(null)','Unexpected dependency')
    return fields

def held(want):
    folder=C.ROOT/'submissions'/want['label'];intent=folder/'INTENT.json';ack=folder/'ACK.json'
    if ack.exists():
        a=C.read(ack);C.require(a['spec']==want,'Conflicting ACK');return a['job']
    if intent.exists():
        old=C.read(intent);C.require(old['spec']==want,'Conflicting intent')
        tag='phase1:'+C.digest(want)
        live=command(['squeue','-u',getpass.getuser(),'--name',want['name'],'-h','-o','%i|%k'])
        past=command(['sacct','-X','-n','-P','-S',old['created'],'--name',want['name'],'--format=JobID%40,Comment%128'])
        ids={s.split('|')[0].strip().split('_')[0] for s in (live+'\n'+past).splitlines() if len(s.split('|'))>=2 and s.split('|')[1].strip()==tag}
        C.require(len(ids)==1 and next(iter(ids)).isdigit(),'Uncertain submission; reconcile, never resubmit')
        job=ids.pop()
    else:
        C.write(intent,dict(spec=want,created=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')))
        cmd=['sbatch','--parsable','--hold','--partition=mit_normal','--account=mit_amf_advanced_cpu','--qos=mit_amf_advanced_cpu',
             '--nodes=1','--ntasks=1','--no-requeue',f"--cpus-per-task={want['cpus']}",f"--mem={want['memory']}M",f"--time={want['minutes']}",
             '--job-name='+want['name'],'--comment=phase1:'+C.digest(want),'--output='+str(C.ROOT/'logs/%x_%A_%a.log')]
        if want['array'] is not None:cmd+=['--array='+want['array']]
        if want['dependency']:cmd+=['--dependency=afterany:'+want['dependency']]
        raw=command(cmd+[want['script'],str(C.ROOT),want['mode']]+want['args'])
        job=raw.split(';')[0];C.require(job.isdigit(),'Uncertain sbatch response')
    C.write(ack,dict(job=job,spec=want));return job

def release(job,want):
    folder=C.ROOT/'submissions'/want['label']
    if (folder/'RELEASED.json').exists():
        C.require(C.read(folder/'RELEASED.json')['job']==job,'Release conflict');return
    raw=command(['scontrol','show','job','-o',job])
    f=check(dict(x.split('=',1) for x in raw.split() if '=' in x),want)
    spool=command(['scontrol','write','batch_script',job,'-'])+'\n'
    C.require(spool== (C.CODE/'job.sh').read_text(),'Spool launcher differs')
    if not (folder/'ALLOCATION.json').exists():C.write(folder/'ALLOCATION.json',f)
    if f.get('Reason')=='JobHeldUser':command(['scontrol','release',job])
    else:C.require(f['JobState'] in ('PENDING','RUNNING','COMPLETING','COMPLETED') and f['Reason'] not in ('JobHeldAdmin','DependencyNeverSatisfied'),'Unexpected release state')
    C.write(folder/'RELEASED.json',dict(job=job,spec_sha256=C.digest(want)))

def pair(label,mode,args,array='0',cpus=2,memory=18432,minutes=120,gate_args=None):
    w=spec(label,mode,args,array=array,cpus=cpus,memory=memory,minutes=minutes);job=held(w)
    if gate_args is not None:
        g=spec(label+'_gate','control',gate_args,dependency=job);follow=held(g);release(follow,g)
    else:follow=None
    release(job,w);return dict(job=job,gate=follow)
