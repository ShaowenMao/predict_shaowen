import tempfile
import unittest
from pathlib import Path
import study


class LineageTests(unittest.TestCase):
    def fixture(self,out):
        source='new_sealed_source'
        m=dict(source=study.PARENT_SOURCE,records=[dict(slot=1,calculation_key='input1'),dict(slot=2,calculation_key='input2')])
        study.write(out/'inputs_lock.json',{})
        study.write(out/'initialized.json',{'passed':True})
        study.write(out/'accepted/slot001.json',dict(source=study.PARENT_SOURCE,slot=1))
        study.write(out/study.RECOVERY/'qualified.json',dict(passed=True,source=source,
            parent_source=study.PARENT_SOURCE,inputs_lock_sha256=study.sha(out/'inputs_lock.json'),
            initialized_sha256=study.sha(out/'initialized.json'),qualification_outputs={},
            preserved_accepted={'slot001.json':study.sha(out/'accepted/slot001.json')}))
        return source,m

    def test_qualified_legacy_and_new_results(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            study.lineage(out,m,source)
            study.result_lineage(out,m['records'][0],study.read(out/'accepted/slot001.json'),source)
            study.write(out/'accepted/slot002.json',dict(source=source,slot=2,
                execution_key=study.execution_key(source,m['records'][1])))
            study.lineage(out,m,source)
            study.result_lineage(out,m['records'][1],study.read(out/'accepted/slot002.json'),source)
            self.assertNotEqual(study.execution_key(source,m['records'][1]),
                                study.execution_key('another_source',m['records'][1]))

    def test_reject_unqualified_legacy(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            study.write(out/'accepted/slot002.json',dict(source=study.PARENT_SOURCE,slot=2))
            with self.assertRaises(AssertionError):study.result_lineage(out,m['records'][1],study.read(out/'accepted/slot002.json'),source)

    def test_reject_changed_legacy(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            with (out/'accepted/slot001.json').open('a') as f:f.write(' ')
            with self.assertRaises(AssertionError):study.lineage(out,m,source)

    def test_reject_wrong_execution_key(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            study.write(out/'accepted/slot002.json',dict(source=source,slot=2,execution_key='wrong'))
            with self.assertRaises(AssertionError):study.result_lineage(out,m['records'][1],study.read(out/'accepted/slot002.json'),source)

    def test_reject_unqualified_source(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            with self.assertRaises(AssertionError):study.lineage(out,m,'other_new_source')
            m['source']='other_old_source'
            with self.assertRaises(AssertionError):study.lineage(out,m,source)

    def test_live_peer_receipt_is_not_scanned(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);source,m=self.fixture(out)
            with (out/'accepted/slot002.json').open('x') as f:f.write('{')
            study.lineage(out,m,source)


if __name__=='__main__':unittest.main()
