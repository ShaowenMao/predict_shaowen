import copy,tomllib,unittest
import common as C,scope
from primary_proof import passing_primary

class PrimaryProof(unittest.TestCase):
    def setUp(self):
        self.key=scope.TARGETS[0];self.row=scope.SCOPE['rows'][self.key]
        self.d=tomllib.loads((C.CODE/'fixtures/096.toml').read_text())
    def test_four_actual_primary_proofs(self):
        for key,row in scope.SCOPE['rows'].items():
            d=tomllib.loads((C.CODE/'fixtures'/(key[-3:]+'.toml')).read_text())
            self.assertEqual(passing_primary(d,row),row['primary_q'])
    def test_wrong_reference_stage(self):
        self.d['failed_stage']='primary'
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_wrong_source(self):
        self.d['source']='bad'
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_wrong_input(self):
        self.d['operator_sha256']='bad'
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_failed_primary(self):
        for a in self.d['attempts']:a['passed']=False
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_ambiguous_primary(self):
        a=copy.deepcopy(next(a for a in self.d['attempts'] if a['passed']));self.d['attempts'].append(a)
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_mass_tolerance_unchanged(self):
        self.d['independent_checks'][0]['independent_mass']=1e-8
        with self.assertRaises(RuntimeError):passing_primary(self.d,self.row)
    def test_authorization_and_launcher(self):
        d=C.read(C.CODE/'AUTHORIZATION.json')
        self.assertTrue(d['approved_by_user'] and d['publication'] and d['allow_core229'])
        self.assertEqual((d['prior_failures_per_core'],d['new_bounded_attempt'],d['reference_attempts_per_core']),(3,4,1))
        self.assertEqual(d['keys'],scope.TARGETS)
        lines=(C.CODE/'job.sh').read_text().splitlines()
        self.assertEqual([s for s in lines if s.startswith('cd ')],['cd /orcd/data/juanes/001/shaowen/'+C.CODE.name])
    def test_recovery_numerics_unchanged(self):
        parent=C.CODE.parent/'phase1_direct_primary8_20261111_v1'
        if not parent.exists():parent=scope.BASE/'phase1_direct_primary8_20261111_v1'
        text=(C.CODE/'recover_flow.jl').read_text().replace('include("SavedPrimary.jl")\n','').replace('SavedPrimary.reaudit','PrecisionRecovery.reaudit')
        self.assertEqual(text,(parent/'recover_flow.jl').read_text())

if __name__=='__main__':unittest.main()
