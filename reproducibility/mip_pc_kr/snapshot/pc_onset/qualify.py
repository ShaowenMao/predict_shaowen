"""Validate relocated launchers against frozen reference configurations."""
import hashlib
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def unpack(name,dest):
    dest.mkdir(exist_ok=True)
    with tarfile.open(ROOT/name) as archive:
        for entry in archive.getmembers():
            target=(dest/entry.name).resolve()
            assert target.is_relative_to(dest.resolve())
            assert entry.isdir() or entry.isfile()
        archive.extractall(dest,filter='data')

def main():
    new=ROOT/'prepared';old=ROOT/'reference_configs'
    unpack('prepared_launchers.tgz',new);unpack('reference_configs.tgz',old)
    matrix=json.loads((new/'matrix.json').read_text());assert len(matrix)==6
    checks=[]
    for r in matrix:
        name=r['name'];ref=name.removesuffix('_pczero')
        a=tomllib.loads((old/'variants'/ref/'campaign.toml').read_text())
        b=tomllib.loads((new/'variants'/name/'campaign.toml').read_text())
        for key in set(a)|set(b):
            if key in ('campaign_id','archive_root','cases','correction','pc_only_experiment'):continue
            assert a[key]==b[key],(name,key)
        assert len(a['cases'])==len(b['cases'])==1
        for key in a['cases'][0]:
            if key not in ('specific_path','specific_sha256','specific_bytes'):
                assert a['cases'][0][key]==b['cases'][0][key],(name,key)
        for key in a['correction']:
            if key!='pc_convention':assert a['correction'][key]==b['correction'][key]
        assert b['correction']['pc_convention']=='zero_before_breakthrough_pc_only'
        assert b['correction']['fault_hysteresis']=='explicit_no_hysteresis'
        assert b['pc_only_experiment']['reference_campaign_sha256']==sha(old/'variants'/ref/'campaign.toml')
        assert len(b['workflow']['retain_years'])==210
        sources=json.loads((new/'variants'/name/'source_comparison.json').read_text())
        changed={x['file'] for x in sources if not x['same_after_launcher_path_relocation']}
        assert changed=={'pair_input_audit.jl','gom_step62_production_final_check.jl'}
        for x in sources:assert sha(new/'variants'/name/'source'/x['file'])==x['new_sha256']
        audit=json.loads((ROOT/'inputs'/ref/'pc_only_audit.json').read_text())
        assert audit['passed'] and audit['output_sha256']==b['cases'][0]['specific_sha256']
        checks.append(dict(name=name,passed=True,scientific_setup_unchanged_except_pc=True,
                           source_changes=sorted(changed)))
    expression='include(ARGS[1]); MipPcOnlyAudit.unit_tests(); '
    expression+='for (root, dirs, files) in walkdir(ARGS[2]); for file in files; endswith(file,".jl") || continue; '
    expression+='e=Meta.parseall(read(joinpath(root,file),String)); '
    expression+='function check(x); x isa Expr || return; x.head in (:error,:incomplete) && error(x); foreach(check,x.args); end; check(e); end; end; '
    expression+='println("ALL_JULIA_SOURCES_PARSE_AND_PC_ONLY_TESTS_PASS")'
    subprocess.run(['julia','--startup-file=no','-e',expression,str(ROOT/'pair_input_audit.jl'),str(new/'variants')],check=True)
    result=dict(passed=True,cases=checks,matrix_sha256=sha(new/'matrix.json'),
        preparation_source_sha256=sha(ROOT/'prepare_remote.py'),
        input_transform_source_sha256=sha(ROOT/'prepare_inputs.py'),
        native_assembled_model_preflight_still_required=True)
    (ROOT/'local_qualification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
