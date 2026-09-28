"""Download and checksum only the small matched property files, not states."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'corrected_hysteresis_15case_20260911_v1'

def main():
    records=json.loads((ROOT/'sources.json').read_text(encoding='utf-8-sig'))
    for r in records:
        name=r['reference']['name']
        dest=ROOT/'reference_inputs'/name
        dest.mkdir(parents=True,exist_ok=True)
        for kind,pin in r['inputs'].items():
            target=dest/(kind+'.mat')
            if not target.exists():
                local=OLD/'curve_visuals_15case_v1/inputs'/(name+'.mat')
                cached=ROOT.parent/'hybrid_pc_breakthrough_6case_20260916_v1/reference_inputs'/name.replace('_mip_','_hybrid_')/(('physical' if kind=='mip' else kind)+'.mat')
                if local.is_file() and kind=='mip' and hashlib.sha256(local.read_bytes()).hexdigest()==pin['sha256']:
                    shutil.copy2(local,target)
                elif cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest()==pin['sha256']:
                    shutil.copy2(cached,target)
                else:subprocess.run(['scp','engaging:'+pin['path'],str(target)],check=True)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==pin['sha256']
        print(name,'downloaded and verified',flush=True)

if __name__=='__main__':main()
