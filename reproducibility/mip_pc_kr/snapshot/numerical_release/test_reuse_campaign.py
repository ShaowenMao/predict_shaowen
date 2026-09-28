from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import types
import unittest
import campaign as C
import reuse_campaign as RC


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(C,'ROOT',self.root/'new'),patch.object(C,'HERE',self.root/'release'),
                      patch.object(C,'S',dict(reuse_parent_root=str(self.root/'old'),
                                             reuse_parent_source_lock='oldlock'))]
        for p in self.patches:p.start()
        C.HERE.mkdir();(C.HERE/'source_lock.json').write_text('{}')
        C.ROOT.mkdir()

    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.tmp.cleanup()

    def test_qualification_required(self):
        with self.assertRaises(FileNotFoundError):RC.require_qualified(C)
        C.write(C.ROOT/'PLAN.json',{})
        C.write(C.ROOT/'REUSE_QUALIFIED.json',dict(passed=True,source_lock='wrong',plan_sha256=C.sha(C.ROOT/'PLAN.json')))
        with self.assertRaises(AssertionError):RC.require_qualified(C)

    def test_plan_binding(self):
        C.write(C.ROOT/'PLAN.json',{})
        C.write(C.ROOT/'REUSE_QUALIFIED.json',dict(passed=True,source_lock=C.sha(C.HERE/'source_lock.json'),
                                               plan_sha256=C.sha(C.ROOT/'PLAN.json')))
        RC.require_qualified(C)
        C.write(C.ROOT/'PLAN.json',dict(changed=True),True)
        with self.assertRaises(AssertionError):RC.require_qualified(C)

    def test_chain_is_afterok_and_sized_from_unique_pilot(self):
        calls=[]
        def submit(*a,**kw):calls.append((a,kw));return str(100+len(calls))
        with patch.object(C,'plan',return_value={'reuse_pilot':{'memory_gib':6}}),patch.object(C,'submit',side_effect=submit):
            RC.launch(C)
        self.assertEqual(calls[0][0][0:5],('reuse-pilot','reuse-pilot',2,6,10800))
        self.assertEqual(calls[1][1]['deps'],['101'])

    def test_shards_parallel_and_collector_depends_on_all(self):
        p={'records':{'a':{'case':'one'},'b':{'case':'two'},'c':{'case':'one'}}}
        calls=[]
        def submit(*a,**kw):calls.append((a,kw));return str(len(calls))
        with patch.object(RC,'parent_plan',return_value=(self.root,p)),patch.object(C,'submit',side_effect=submit):
            RC.start_scan(C)
        self.assertEqual(len(calls),3)
        self.assertTrue(all(c[0][2:4]==(1,2) for c in calls[:2]))
        self.assertEqual(calls[-1][1]['deps'],['1','2'])

    def test_completed_serial_fingerprints_preserved(self):
        C.S['reuse_parent_plan_sha256']='parent';records={'a':{'x':1}}
        path=self.root/'index.jsonl'
        d=dict(parent_plan='parent',method='m',key='a',record=C.digest(records['a']),fingerprint={'key':'f'})
        path.write_text(json.dumps(d)+'\n'+ '{"incomplete":')
        with patch.object(RC,'method',return_value='m'):
            self.assertEqual(RC.read_fingerprints(C,path,records,True),{'a':{'key':'f'}})
            with self.assertRaises(ValueError):RC.read_fingerprints(C,path,records)

    def test_fingerprint_journal_conflict_rejected(self):
        C.S['reuse_parent_plan_sha256']='parent';records={'a':{'x':1}}
        path=self.root/'index.jsonl'
        d=dict(parent_plan='parent',method='m',key='a',record='wrong',fingerprint={'key':'f'})
        path.write_text(json.dumps(d)+'\n')
        with patch.object(RC,'method',return_value='m'):
            with self.assertRaises(AssertionError):RC.read_fingerprints(C,path,records)

    def test_verify_all_members_not_only_owner(self):
        p=dict(records={'a':dict(numerical_key='n',placements=['a','b'])},
               placements={'a':dict(key='a'),'b':dict(key='b')},groups={'n':{'fingerprint':{'n':1}}})
        with patch.object(RC,'fingerprint',side_effect=[{'n':1},{'n':2}]) as fp:
            with self.assertRaises(AssertionError):RC.verify_group(C,p,'a')
            self.assertEqual(fp.call_count,2)

    def test_worker_calls_unchanged_compute_once_and_maps_owner(self):
        r=dict(key='a',resource=dict(memory_gib=4))
        done=dict(native_passed=True,memory={'passed':True})
        ctx=types.SimpleNamespace(memory_snapshot=lambda:{'passed':True})
        with patch.object(C,'bound',return_value=dict(memory_gib=4,cpus=2)),patch.object(C,'pins'), \
             patch.object(C,'plan',return_value={'records':{'a':r}}),patch.object(C,'context',return_value=ctx), \
             patch.object(RC,'require_qualified'),patch.object(RC,'verify_group'), \
             patch.object(C,'compute',return_value=done) as compute,patch.dict(os.environ,SLURM_JOB_ID='123'):
            C.worker('core_a','a')
            self.assertEqual(compute.call_count,1)
            self.assertEqual(C.read(C.accepted_path('a')),done)

    def test_no_compute_without_qualification(self):
        r=dict(key='a',resource=dict(memory_gib=4))
        ctx=types.SimpleNamespace(memory_snapshot=lambda:{'passed':True})
        with patch.object(C,'bound',return_value=dict(memory_gib=4,cpus=2)),patch.object(C,'pins'), \
             patch.object(C,'plan',return_value={'records':{'a':r}}),patch.object(C,'context',return_value=ctx), \
             patch.object(C,'compute') as compute,patch.dict(os.environ,SLURM_JOB_ID='123'):
            with self.assertRaises(FileNotFoundError):C.worker('core_a','a')
            compute.assert_not_called()
            self.assertFalse(C.accepted_path('a').exists())


if __name__=='__main__':unittest.main()
