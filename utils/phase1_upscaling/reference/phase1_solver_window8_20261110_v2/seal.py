import common as C
names=['common.py','slurm.py','window.py','test_window.py','job.sh','seal.py']
C.write(C.CODE/'LOCK.json',dict(code={n:C.sha(C.CODE/n) for n in names},worker_source='846e9f7ac692a9da5df92254df6c0b5f8a0a2c3ac35a5507d86a4429fd2dd502',max_workers=64,max_groups_equivalent=8,scientific_changes=False,worker_changes=False,publication_changes=False,automatic_core_retry=False,finite_batches=20,prior_operational_failed_launches=1,failed_source='4ac5b8a073c7a89968c60c23442d41894878d5362028b28c078e6919f389ecf9',repair='Pass immutable spec root to actual launcher; compute cross-root proof and terminal no-science audit before one corrected pilot'))
print(C.sha(C.CODE/'LOCK.json'))
