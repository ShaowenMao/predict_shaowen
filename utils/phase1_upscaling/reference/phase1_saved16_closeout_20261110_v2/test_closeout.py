import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
from closeout import promote_archive,verify_case,check_preserved,METHOD,CONFIG


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.record=dict(case_id='s01_c007_pc04_ca40_IND_003',slot=4,configuration_sha256=CONFIG)
        self.adoption=dict(policy='fixture',new_solves=0)

    def tearDown(self):self.temp.cleanup()

    def archive(self):
        work=self.root/'work';work.mkdir();C.write(work/'EXPORT_ADOPTION.json',self.adoption)
        C.write(work/'data.json',dict(saved=True))
        path=self.root/'private.zip';C.publish(work,self.record,METHOD,path)
        return path

    def test_publish_and_verified_resume_never_repackages(self):
        source=self.archive();target=self.root/'accepted/core004.zip'
        receipt=promote_archive(source,target,self.record,self.adoption)
        before=target.stat().st_mtime_ns
        with patch.object(C,'publish',side_effect=AssertionError('No recalculation/repack')):
            same=promote_archive(source,target,self.record,self.adoption)
        self.assertEqual(receipt,same);self.assertEqual(before,target.stat().st_mtime_ns)

    def test_foreign_existing_result_not_overwritten(self):
        source=self.archive();target=self.root/'accepted.zip';target.write_bytes(b'other')
        with self.assertRaises(RuntimeError):promote_archive(source,target,self.record,self.adoption)
        self.assertEqual(target.read_bytes(),b'other')

    def test_wrong_adoption_rejected(self):
        source=self.archive();target=self.root/'accepted.zip'
        with self.assertRaises(RuntimeError):promote_archive(source,target,self.record,dict(policy='different'))
        self.assertFalse(target.exists())

    def test_wrong_record_rejected(self):
        source=self.archive();record=dict(self.record,slot=5)
        with self.assertRaises(RuntimeError):promote_archive(source,self.root/'accepted.zip',record,self.adoption)

    def test_corrupt_archive_rejected(self):
        source=self.archive();source.write_bytes(b'corrupt')
        with self.assertRaises(Exception):promote_archive(source,self.root/'accepted.zip',self.record,self.adoption)
        self.assertFalse((self.root/'accepted.zip').exists())

    def test_missing_case_position_rejected(self):
        case=self.record['case_id'];C.write(self.root/'plans'/(case+'.json'),dict(records=[self.record]))
        with self.assertRaises(RuntimeError):verify_case(self.root,case)

    def test_duplicate_case_positions_rejected(self):
        case=self.record['case_id'];C.write(self.root/'plans'/(case+'.json'),dict(records=[self.record]*522))
        with self.assertRaises(RuntimeError):verify_case(self.root,case)

    def test_changed_evidence_rejected(self):
        p=self.root/'evidence.json';C.write(p,dict(saved=True));state=dict(files={str(p):C.sha(p)},archives={})
        check_preserved(state);p.write_text('changed')
        with self.assertRaises(RuntimeError):check_preserved(state)

    def test_mount_device_number_is_node_local(self):
        state=dict(files={},archives={'/shared/accepted.zip':[99,12345,20570,1793411535349927946]})
        with patch('closeout.fingerprints',return_value=[2097259,12345,20570,1793411535349927946]):
            check_preserved(state)

    def test_portable_file_changes_still_rejected(self):
        old=[99,12345,20570,1793411535349927946]
        state=dict(files={},archives={'/shared/accepted.zip':old})
        for field in (1,2,3):
            new=old.copy();new[0]=2097259;new[field]+=1
            with self.subTest(field=field),patch('closeout.fingerprints',return_value=new),self.assertRaises(RuntimeError):
                check_preserved(state)


if __name__=='__main__':unittest.main()
