"""Explicit small-batch submission using the proven held/ACK/release helper."""
import argparse
import os
import shutil
import common as C
import scope
import slurm as S
import publish as P

def admitted_keys(number):
    keys=scope.batch_keys(number)
    if number:
        pilot=C.read(C.ROOT/'batches/000/PASS.json')
        C.require(pilot['all_passed'] and pilot['published']==scope.FIRST and pilot['source']==C.sha(C.CODE/'LOCK.json'),'First three must pass before expansion')
    if number>1:
        previous=C.read(C.ROOT/'batches'/f'{number-1:03d}'/'PASS.json')
        C.require(previous['passed'] and previous['source']==C.sha(C.CODE/'LOCK.json'),'Previous bounded batch unresolved')
    return keys

def run(number):
    C.require(C.ROOT==C.CODE/'output','Private output required');C.verify_source();P.stopped()
    keys=admitted_keys(number);C.require(1<=len(keys)<=8,'Batch size')
    C.require(shutil.disk_usage(C.ROOT).free>20*1024**3,'Shared disk headroom')
    (C.ROOT/'logs').mkdir(exist_ok=True)
    with C.locked(C.ROOT/'submit.lock'):
        folder=C.ROOT/'batches'/f'{number:03d}'
        admission=dict(keys=keys,source=C.sha(C.CODE/'LOCK.json'),automatic_retry=False)
        C.write(folder/'ADMITTED.json',admission)
        w=S.spec(f'targeted{number:03d}','worker',[str(number)],array=f'0-{len(keys)-1}',cpus=2,memory=18432,minutes=120)
        job=S.held(w)
        g=S.spec(f'targeted{number:03d}_publish','publish',[str(number)],dependency=job,cpus=1,memory=2048,minutes=30)
        follow=S.held(g)
        S.release(follow,g);S.release(job,w)
        C.write(folder/'LAUNCHED.json',dict(worker=job,publish=follow,keys=keys,source=C.sha(C.CODE/'LOCK.json')))
        print(C.read(folder/'LAUNCHED.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('batch',type=int);run(p.parse_args().batch)
