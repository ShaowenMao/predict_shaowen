import tempfile
import unittest
from pathlib import Path
import study


class RecoveryScopeTests(unittest.TestCase):
    def test_exact_recovery_scope(self):
        self.assertEqual(len(study.RECOVERY_SLOTS),29)
        self.assertEqual(len(set(study.RECOVERY_SLOTS)),29)
        self.assertTrue(all(1<=x<=522 for x in study.RECOVERY_SLOTS))
        self.assertTrue(set(study.RECOVERY_SLOTS).isdisjoint({32,86,118,179,295,353,460,473}))

    def test_original_success_cannot_be_recomputed(self):
        with self.assertRaisesRegex(AssertionError,'originally accepted'):
            study.slot_run(Path('/not_used'),{},32)

    def test_endpointfix_source_needs_preserved_receipt(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);r={'slot':7,'calculation_key':'a'};source='new'
            study.write(out/'accepted/slot007.json',{'source':study.PREVIOUS_SOURCE})
            study.write(out/study.RECOVERY/'qualified.json',dict(source=source,passed=True,
                preserved_accepted={'slot007.json':study.sha(out/'accepted/slot007.json')}))
            study.result_lineage(out,r,study.read(out/'accepted/slot007.json'),source)
            with self.assertRaises(AssertionError):
                study.result_lineage(out,{'slot':11},study.read(out/'accepted/slot007.json'),source)


if __name__=='__main__':unittest.main()
