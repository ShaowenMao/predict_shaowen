"""Submit only this isolated study. Record every submitted job immediately."""
import subprocess
import study


def submit(label, args):
    path = study.OUT/f'submissions/{label}.json'
    assert not path.exists(), ('Already submitted; inspect rather than duplicate', label)
    command = ['sbatch', '--parsable', f'--export=ALL,COMPARISON_SOURCE={study.SOURCE}',
               f'--output={study.OUT}/logs/%x_%A_%a.out', f'--error={study.OUT}/logs/%x_%A_%a.err'] + args
    result = subprocess.check_output(command, text=True).strip()
    job = result.split(';')[0]
    assert job.isdigit(), result
    study.write(path, dict(job=job, command=command))
    print(label, job, flush=True)
    return job


if __name__ == '__main__':
    manifest = study.verify()
    job_script = str(study.SOURCE/'job.sh')
    qualification = submit('qualification', ['--job-name=cmp3_qualify', '--time=02:00:00', job_script, 'qualify'])
    for _, case, _ in study.CASES:
        if case == 's03_c022_case12':
            continue
        c = manifest['cases'][case]
        pilot = submit(case+'_pilot', [f'--dependency=afterok:{qualification}',
                       f'--array=1-{len(c["pilots"])}', '--job-name=cmp3_pilot', job_script, 'pilot', case])
        gate = submit(case+'_gate', [f'--dependency=afterok:{pilot}', '--time=01:00:00',
                      '--job-name=cmp3_gate', job_script, 'gate', case])
        main = submit(case+'_main', [f'--dependency=afterok:{gate}',
                      f'--array=1-{len(c["main_batches"])}', '--job-name=cmp3_main', job_script, 'main', case])
        submit(case+'_assembly', [f'--dependency=afterany:{main}', '--time=02:00:00',
               '--job-name=cmp3_assemble', job_script, 'finish', case])
