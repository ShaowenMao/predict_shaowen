import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dispatch as D
from common import write, sha
import bounded_io
import reuse


class Admission(unittest.TestCase):
    def row(self, job='1',state='RUNNING',cpu='8',mem='18G',reason='None'):
        return [job,state,cpu,mem,reason,D.ACCOUNT,D.ACCOUNT]

    def test_running_and_eligible_pending_count(self):
        rows=[self.row(),self.row('2','PENDING','8','24G','Resources')]
        r=D.admission(rows,set(),2,512)
        self.assertEqual((r['used_cpus'],r['used_mib']),(16,42*1024))

    def test_own_blocked_pending_counts(self):
        r=D.admission([self.row('2','PENDING','8','24G','Dependency')],{'2'},1,512)
        self.assertEqual(r['used_cpus'],8)

    def test_external_held_explicitly_recorded(self):
        rows=[self.row('2','PENDING','8','24G','Dependency'),self.row('3','PENDING','1','2G','JobHeldUser')]
        r=D.admission(rows,set(),1,512)
        self.assertEqual(r['used_cpus'],0)
        self.assertEqual(len(r['external_dependency_or_user_held']),2)

    def test_cpu_cap(self):
        self.assertFalse(D.admission([self.row(cpu='500')],set(),1,512)['passed'])

    def test_memory_cap(self):
        self.assertFalse(D.admission([self.row(mem='1984G')],set(),1,512)['passed'])

    def test_bad_state_fails_closed(self):
        with self.assertRaises(AssertionError):D.admission([self.row(state='UNKNOWN')],set(),1,512)

    def test_bad_qos_fails_closed(self):
        r=self.row();r[-1]='normal'
        with self.assertRaises(AssertionError):D.admission([r],set(),1,512)

    def test_memory_parse(self):
        for value,expected in [('1.25G',1280),('512M',512),('2T',2097152),('1024K',1)]:
            self.assertEqual(D.memory_mib(value),expected)
        with self.assertRaises(ValueError):D.memory_mib('garbage')

    def test_duplicate_queue_fails_closed(self):
        with self.assertRaises(AssertionError):D.admission([self.row(),self.row()],set(),1,512)


class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'LOCK.json').write_text('{}')
        self.patch=patch.object(D,'HERE',self.root);self.patch.start()

    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def intent(self):write(self.root/'intents/x.intent.json',dict(token=D.token('x')))

    def test_no_intent(self):self.assertIsNone(D.returned_job('x',[]))

    def test_ambiguous_never_resubmits(self):
        self.intent()
        for queue in ([],[['123',D.token('x')],['456',D.token('x')]]):
            with self.assertRaises(AssertionError):D.returned_job('x',queue)

    def test_unique_scheduler_reconciliation(self):
        self.intent()
        self.assertEqual(D.returned_job('x',[['123',D.token('x')]]),'123')
        self.assertEqual(D.returned_job('x',[]),'123')

    def test_wrong_token_held(self):
        write(self.root/'intents/x.returned.json',dict(job='123',token='wrong'))
        with self.assertRaises(AssertionError):D.returned_job('x',[])

    def test_immutable_publication(self):
        p=self.root/'accepted.json';write(p,dict(passed=True));h=sha(p)
        write(p,dict(passed=True));self.assertEqual(h,sha(p))
        with self.assertRaises(AssertionError):write(p,dict(passed=False))

    @unittest.skipIf(os.name=='nt','Live Linux lock checked on Engaging')
    def test_single_writer(self):
        with D.single_writer(self.root/'lock'):
            with self.assertRaises(BlockingIOError):
                with D.single_writer(self.root/'lock'):pass

    def test_bounded_before_audit(self):
        d=D.dag();self.assertEqual(set(d),{'audit','warm'})
        self.assertTrue(all(p['cpus']<=2 for p in d.values()))

    def test_bounded_reader_exact(self):
        p=self.root/'input.bin'
        p.write_bytes(bytes(range(256))*10001)
        for head,edits in [(96,[(8,16,bytes(8))]),(104,[(8,16,bytes(8)),(72,104,bytes(32))])]:
            self.assertEqual(reuse.normalized_file(p,head,edits),bounded_io.normalized_file(p,head,edits))

    def test_scientific_and_file_hash_are_distinct(self):
        p=self.root/'flows.jsonl';q=self.root/'flows2.jsonl'
        p.write_text(json.dumps({'value':2,'info':{'seconds':3,'residual':1e-12}})+'\n')
        q.write_text(json.dumps({'value':2,'info':{'seconds':6,'residual':1e-12}})+'\n')
        self.assertNotEqual(sha(p),sha(q))
        self.assertEqual(bounded_io.scientific_hash(p),bounded_io.scientific_hash(q))
        q.write_text(json.dumps({'value':2,'info':{'seconds':6,'residual':1e-4}})+'\n')
        self.assertNotEqual(bounded_io.scientific_hash(p),bounded_io.scientific_hash(q))


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__)))
    if not result.wasSuccessful():raise SystemExit(1)
    if os.name!='nt':
        write(Path(__file__).parent/'tests.json',dict(passed=True,tests=result.testsRun,skipped=len(result.skipped),
                                                  platform=os.name,full_campaign_launched=False))
