"""Initialize and arm one finite qualification chain; no production transition."""
import argparse
import shutil
import common as C
import slurm as S
from packet_qualification import LABELS,original_unchanged

def main(mode):
    lock=C.verify_source();case=C.read(C.CODE/'STAGED.json')['pilot_cases'][0]
    C.ROOT=C.CODE/'numerical'
    if mode=='initialize':
        C.require(not (C.ROOT/'RUN.json').exists(),'Qualification already initialized')
        original_unchanged()
        (C.ROOT/'logs').mkdir(parents=True);(C.ROOT/'plans').mkdir()
        C.write(C.ROOT/'RUN.json',dict(code=str(C.CODE),source=C.sha(C.CODE/'LOCK.json')))
        C.write(C.ROOT/'SCOPE.json',dict(purpose='qualification',cases=[case],packet_count=1))
        C.write(C.ROOT/'STATE.json',dict(cursor=0,active=[]))
        shutil.copyfile(C.CODE/'plans'/f'{case}.json',C.ROOT/'plans'/f'{case}.json')
        C.require(C.sha(C.ROOT/'plans'/f'{case}.json')==lock['plans'][case],'Plan copy mismatch')
        return
    C.require(mode=='submit','Unknown mode')
    original_unchanged()
    predecessor=None;chain=[]
    for stage,label in LABELS.items():
        want=S.spec(label,'integration',[stage],array=None if stage=='unit' else '0',dependency=predecessor,
                    cpus=1 if stage=='unit' else 2,memory=1024 if stage=='unit' else 18432,
                    minutes=10 if stage in ('unit','audit') else 120)
        job=S.held(want);S.release(job,want);chain.append(dict(stage=stage,job=job));predecessor=job
    C.write(C.ROOT/'QUALIFICATION_CHAIN.json',dict(jobs=chain,source=C.sha(C.CODE/'LOCK.json'),production_modified=False))
    print(chain,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','submit']);main(p.parse_args().mode)
