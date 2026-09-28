"""Consolidated stage launcher: upscaler 1.12, reservoir 1.10, short MATLAB paths."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

from common import HERE,read,write,sha,pins_ok,verify
from resource import cgroup,snapshot,passed,parse_allocation
import stages


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def matlab_environment(cfg):
    old=Path(cfg['old_root'])/'releases'/cfg['old_release'];source=old/'temp_paths.py'
    assert sha(source)==read(old/'source_lock.json')['files']['temp_paths.py']
    module=load_module('qualified_temp_environment',source)
    job=os.environ['SLURM_JOB_ID'];lock=sha(HERE/'LOCK.json')[:16]
    paths=module.create(module.TEMP_ROOT,job,lock)
    os.environ.update({k:str(paths['tmp']) for k in ('TMPDIR','TMP','TEMP','TEMPDIR')})
    os.environ.update(MATLAB_PREFDIR=str(paths['preferences']),MCR_CACHE_ROOT=str(paths['cache']))
    module.validate_environment(module.TEMP_ROOT,job,lock,os.environ)


def main(stage,julia):
    cfg=verify();lock=read(HERE/'QUALIFIED_LAUNCHER_LOCK.json')
    assert lock['base_lock_sha256']==sha(HERE/'LOCK.json')
    pins_ok({HERE/n:h for n,h in lock['files'].items()})
    version=subprocess.check_output([julia,'--version'],text=True).strip()
    expected='1.10.4' if stage in ('fullgrid','probe-reservoir') else '1.12.6'
    assert version=='julia version '+expected,version
    if stage in ('assemble','probe-upscaling'):
        matlab_environment(cfg)
    if stage.startswith('probe-'):
        assert stage in ('probe-upscaling','probe-reservoir')
        cpu,mib=(2,3072) if stage=='probe-upscaling' else (1,512)
        actual=subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
        parse_allocation(actual,cpu,mib);cg=cgroup(os.environ['SLURM_JOB_ID'])
        out=HERE/'launcher_probes'/stage;out.mkdir(parents=True,exist_ok=False)
        if stage=='probe-upscaling':
            stages.matlab(cfg,'verify_matlab_startup',[out/'matlab.json'],out/'matlab.log')
            assert read(out/'matlab.json')['passed']
        mem=snapshot(cg);assert passed(mem,mib)
        write(out/'PASS.json',dict(passed=True,version=version,job=os.environ['SLURM_JOB_ID'],
                                   memory=mem,actual_scontrol=actual,launcher_lock_sha256=sha(HERE/'QUALIFIED_LAUNCHER_LOCK.json'),
                                   no_numerical_work_repeated=True))
        return
    if stage=='fullgrid':
        module=load_module('qualified_production_runtime',HERE/'repairs/runtime_version_v1/recover.py')
        sys.argv[1]=julia
        stages.fullgrid=module.run_runtime
    stages.run(stage)


if __name__=='__main__':main(sys.argv[1],sys.argv[2])
