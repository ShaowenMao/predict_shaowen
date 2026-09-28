"""Read-only inventory of matched MIP and archived Pc convention inputs."""
import hashlib
import json
import shlex
from pathlib import Path

BASE=Path('/orcd/data/juanes/001/shaowen/predict_shaowen/candidates/cell_pc_mip_kr_v1_20260908')
OLD=BASE/'corrected_hysteresis_15case_20260911_v1'
MATRIX=BASE/'comparative_3case_wrr30_60_pc_conventions_20260910_v1/reservoir_matrix_launch_20260911_v1'

def env(path):
    return {s[7:].split('=',1)[0]:shlex.split(s.split('=',1)[1])[0]
            for s in path.read_text().splitlines() if s.startswith('export ')}

def main():
    records=[]
    for r in json.loads((OLD/'submissions.json').read_text()):
        if r['workflow']!='mip':continue
        root=Path(r['root']);h=env(root/'locked.env')
        if r['risk']=='intermediate' and r['angle']==30:
            donor=BASE/'reservoir_pc_breakthrough_s03_c022_case12_20260910_v1'
            z=env(donor/'locked.env')
            physical=BASE/'reservoir_comparison_s03_c022_case12_20260910_v1'
            p=env(physical/'locked.env')
        else:
            donor=MATRIX/'variants'/f"{r['case']}_wrr{r['angle']}_zero_before_breakthrough"
            z=env(donor/'locked.env')
            physical=MATRIX/'variants'/f"{r['case']}_wrr{r['angle']}_physical_prebreakthrough"
            p=env(physical/'locked.env')
        paths={'mip':h['GOM_PRODUCTION_SPECIFIC_MAT'],
               'physical':p['GOM_PRODUCTION_SPECIFIC_MAT'],
               'zero':z['GOM_PRODUCTION_SPECIFIC_MAT']}
        pins={}
        for kind,path in paths.items():
            source=Path(path)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            expected={'mip':h,'physical':p,'zero':z}[kind]['GOM_PRODUCTION_SPECIFIC_SHA256']
            assert digest==expected,(kind,path)
            pins[kind]={'path':path,'sha256':digest,'bytes':source.stat().st_size}
        records.append(dict(reference=r,inputs=pins,zero_launcher=str(donor),physical_launcher=str(physical)))
    assert len(records)==6
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
