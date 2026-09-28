"""Prove dispatcher semantics unchanged; freeze the corrected stage routing."""
from common import HERE,sha,write

original=(HERE/'dispatch.py').read_text()
qualified=(HERE/'qualified_dispatch.py').read_text()
old="str(HERE/'job.sh')";new="str(HERE/'qualified_job.sh')"
assert original.count(old)==qualified.count(new)==1
guard="""    from common import pins_ok
    guard=read(HERE/'QUALIFIED_LAUNCHER_LOCK.json')
    assert guard['base_lock_sha256']==sha(HERE/'LOCK.json')
    pins_ok({HERE/n:h for n,h in guard['files'].items()})
"""
assert qualified.count(guard)==1
assert qualified.replace(new,old).replace(guard,'')==original
names=['qualified_dispatch.py','qualified_driver.py','qualified_job.sh','repairs/runtime_version_v1/recover.py']
write(HERE/'QUALIFIED_LAUNCHER_LOCK.json',dict(base_lock_sha256=sha(HERE/'LOCK.json'),
      files={n:sha(HERE/n) for n in names},dispatcher_changes=['job_entry_path','additional_source_guard'],
      source_of_matlab_environment='same frozen temp_paths.py used by passed assembly recovery',
      source_of_fullgrid_handler='same runtime_version_v1 recovery used in full-grid qualification',
      upscaler_julia='1.12.6',reservoir_julia='1.10.4',old_campaign_modified=False))
print('DISPATCHER_SOURCE_EQUIVALENCE_PASS')
