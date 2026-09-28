"""One case-angle chain; submit four independent chains without an array cap."""
import os
import subprocess
from study import OUT,SOURCE,CASE,ANGLE,verify,write

verify()
assert not (OUT/'submissions').exists()
(OUT/'submissions').mkdir()
(OUT/'logs').mkdir(exist_ok=True)


def submit(label,mode,deps=None,array=None,mem='32G',hours=12,dependency='afterok'):
    args=['sbatch','--parsable',f'--job-name=hyb{ANGLE}_{CASE[:8]}_'+label,
          '--exclude=node1614,node1617','--mem='+mem,'--time='+str(hours)+':00:00',
          '--output='+str(OUT/'logs'/'%x-%A_%a.out'),'--error='+str(OUT/'logs'/'%x-%A_%a.err'),
          '--chdir='+str(OUT)]
    if deps:args+=['--dependency='+dependency+':'+':'.join(deps)]
    if array:args+=['--array='+array]
    args += [str(SOURCE/'job.sh'),mode]
    raw=subprocess.check_output(args,text=True).strip()
    job=raw.split(';')[0];assert job.isdigit(),raw
    write(OUT/'submissions'/f'{label}.json',dict(job=job,args=args))
    print(CASE,ANGLE,label,job,flush=True)
    return job


upstream=os.environ.get('HYBRID_UPSTREAM_JOBS','')
dependencies=upstream.split(':') if upstream else None
assert not dependencies or all(j.isdigit() for j in dependencies)
# Wait for writers to exit even if their unrelated static Kr solve failed.
# Preparation still requires a validated planned Pc input for all 522 cores.
p=submit('prepare','prepare',dependencies,mem='18G',hours=4,dependency='afterany')
w=submit('dynamic','window',[p],array='1-6',mem='32G',hours=24)
a=submit('assemble','assemble',[w],mem='18G',hours=2)
q=submit('qualify','qualify',[a],mem='18G',hours=3)
l=submit('launch','launch',[q],mem='2G',hours=1)
write(OUT/'SUBMITTED.json',dict(case=CASE,angle=ANGLE,prepare=p,windows=w,assemble=a,
      qualify=q,launch=l,full_reservoir_submitted=False,
      full_reservoir_submitted_by_qualified_launcher=True))
