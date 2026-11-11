"""Revalidate only Core 3, reusing its immutable MATLAB reference and raw input."""
import argparse
import os
from pathlib import Path
import socket
import time
import study

OLD_SOURCE=study.BASE/'releases/julia_pc_938e58bbdf2c'
OLD_HASH='938e58bbdf2c09edbd8b2d0590dacdd8d596d4c946c4ff9af0df571e58e74363'
OLD_OUTPUT=study.BASE/'julia_pc_module_20260909_v1_loadfix1'
FAILED=OLD_OUTPUT/'attempts/core003_job22392978_1788971302954767071'


def verify_numerics():
    lock=study.read(OLD_SOURCE/'source_lock.json')
    assert lock['sha256']==OLD_HASH and study.identity(lock['files'])==OLD_HASH
    for name,digest in lock['files'].items():
        assert study.sha(OLD_SOURCE/name)==digest
        if name.startswith(('src/','vendor/')) or name in ('run_pc.jl','Project.toml'):
            assert study.sha(study.ROOT/name)==digest,'Numerical runtime changed'


def prepare(out):
    assert study.BASE in out.resolve().parents and not out.exists()
    verify_numerics()
    previous=study.read(OLD_OUTPUT/'inputs/manifest.json')
    assert previous['source']==OLD_HASH
    receipts={}
    for core in [7,18,22,25,47]:
        paths=list((OLD_OUTPUT/'attempts').glob(f'core{core:03d}_*/completion.json'))
        assert len(paths)==1 and study.read(paths[0])['all_equivalence_gates_passed']
        receipts[str(paths[0])]=study.sha(paths[0])
    assert not (FAILED/'completion.json').exists()
    assert study.read(FAILED/'julia/mismatch.json')['completed_points']==100
    for name in ['input_receipt.json','execution_error.json','julia/mismatch.json']:
        receipts[str(FAILED/name)]=study.sha(FAILED/name)
    (out/'inputs').mkdir(parents=True,exist_ok=False)
    study.write(out/'inputs/manifest.json',dict(source=study.verify(),core=3,
        original_source=OLD_HASH,failed_attempt=str(FAILED),protected_receipts=receipts,
        entry=next(x for x in previous['cores'] if x['core']==3),
        exports=study.read(FAILED/'input_receipt.json'),rerun_cores=[3],
        numerical_runtime_unchanged=True,flow_solves=False,reservoir_simulations=False,
        production_promotion=False))


def run(out):
    assert os.environ.get('SLURM_JOB_ID'),'Compute-node execution only'
    assert study.BASE in out.resolve().parents
    source=study.verify();manifest_path=out/'inputs/manifest.json'
    manifest_hash=study.sha(manifest_path);m=study.read(manifest_path)
    assert m['source']==source and m['rerun_cores']==[3]
    attempt=out/'attempts'/f'core003_job{os.environ["SLURM_JOB_ID"]}_{time.time_ns()}'
    attempt.mkdir(parents=True,exist_ok=False)
    print('ATTEMPT',attempt,flush=True)
    prep=FAILED/'preparation'
    def protect():
        verify_numerics()
        for name,digest in m['protected_receipts'].items():assert study.sha(name)==digest
        for name,digest in m['exports'].items():assert study.sha(prep/name)==digest
        assert study.sha(manifest_path)==manifest_hash and study.verify()==source
    try:
        protect();old=study.frozen();entry=m['entry'];item=entry['item']
        plan,deps=old.upstream(item);numeric=plan['record']['spec']['numeric_package']
        study.write(attempt/'provenance.json',dict(source=source,manifest_sha256=manifest_hash,
            node=socket.gethostname(),dependencies=deps,original_attempt=str(FAILED),
            original_inputs_reused=True,numerical_runtime_unchanged=True,
            slurm={k:v for k,v in os.environ.items() if k.startswith('SLURM_')}))
        v=study.ROOT/'validation'
        study.command(attempt/'unit_tests','julia',['julia','--startup-file=no','--math-mode=ieee',
            '--check-bounds=yes',str(study.ROOT/'test/runtests.jl')])
        study.command(attempt/'event_tests','julia',['julia','--startup-file=no','--math-mode=ieee',
            '--check-bounds=yes',str(v/'test_event_audit.jl')])
        study.command(attempt/'julia_process','julia',['julia','--startup-file=no','--math-mode=ieee',
            str(v/'validate.jl'),'core_recovery',str(prep),str(attempt/'julia')])
        result=study.read(attempt/'julia/comparison.json')
        assert result['all_equivalence_gates_passed'] and result['same_model_replay_bitwise_passed']
        assert result['points']==100 and len(result['inactive_endpoint_audits'])==1
        assert not result['cross_model_event_bins_exact']
        quote=old.quote
        expr=(f'addpath({quote(v)});check_archive({quote(numeric)},{quote(prep/"model.bin")},'
              f'{quote(attempt/"julia")},{quote(m["exports"]["raw.bin"])},'
              f'{quote(attempt/"archive_check/pass.json")});')
        study.command(attempt/'archive_check','matlab',['matlab','-batch',expr])
        assert study.read(attempt/'archive_check/pass.json')['streaming_matlab_consumer_passed']
        accuracy=Path(entry['accuracy_file']);assert study.sha(accuracy)==entry['accuracy_sha256']
        study.write(attempt/'sampling_accuracy.json',study.read(accuracy))
        _,after=old.upstream(item);assert after['sha256']==deps['sha256'];protect()
        study.write(attempt/'completion.json',dict(core=3,source=source,execution_complete=True,
            numerical_runtime_unchanged=True,previous_outputs_unchanged=True,
            all_equivalence_gates_passed=True,same_model_replay_bitwise_passed=True,
            cross_model_event_bins_exact=False,inactive_endpoint_audits=result['inactive_endpoint_audits'],
            native_archive_validated_in_julia_and_matlab=True,reference_reused=True,
            production_promotion=False,reservoir_export_approved=False,
            outputs={p.relative_to(attempt).as_posix():study.sha(p) for p in sorted(attempt.rglob('*')) if p.is_file()}))
        print('COMPLETE Core3 unchanged numerics; full state and same-model replay verified',flush=True)
    except Exception as err:
        study.write(attempt/'execution_error.json',dict(error=str(err)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('output',type=Path)
    a=p.parse_args();prepare(a.output) if a.mode=='prepare' else run(a.output)
