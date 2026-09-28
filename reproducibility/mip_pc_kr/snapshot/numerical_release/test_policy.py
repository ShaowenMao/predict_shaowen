import copy,unittest
from policy import route,canaries,room,claim_action,gate_status,seconds_allowed

def feature(n=30000,c=.5):
    return dict(num_cells=n,cartesian_dimensions=[100,n//10000,100],clay_cell_fraction=c,
                log10_permeability_span=[7,7,7])

class Tests(unittest.TestCase):
    def test_small_not_oversized(self):self.assertEqual(route(feature(),8)[0]['memory_gib'],4)
    def test_large_mixed(self):self.assertEqual(route(feature(150000,.27),8)[0]['memory_gib'],24)
    def test_homogeneous(self):self.assertEqual(route(feature(120000,1),8)[0]['memory_gib'],8)
    def test_oversized_hold(self):self.assertIsNone(route(feature(160000),8)[0])
    def test_contrast_hold(self):self.assertIsNone(route(feature(),6)[0])
    def test_geometry_hold(self):
        f=feature();f['cartesian_dimensions']=[100,3,99]
        self.assertIsNone(route(f,8)[0])
    def test_immutability(self):
        f=feature();old=copy.deepcopy(f);route(f,8);self.assertEqual(f,old)
    def test_boundary(self):
        self.assertTrue(room(498,1980,2,4));self.assertFalse(room(499,0,2,4));self.assertFalse(room(0,1981,2,4))
    def test_claims(self):
        self.assertEqual(claim_action(None),'SUBMIT')
        self.assertEqual(claim_action({}),'HOLD_UNCERTAIN')
        self.assertEqual(claim_action({},'123'),'RECONCILE')
        self.assertEqual(claim_action({'job':'123'}),'PRESERVE')
    def test_failed_gate(self):self.assertEqual(gate_status(['a'],{},['a']),'HOLD')
    def test_wait_gate(self):self.assertEqual(gate_status(['a'],{},[]),'WAIT')
    def test_native_gate(self):
        a={'a':dict(memory={'passed':True},native_passed=False)}
        self.assertEqual(gate_status(['a'],a,[]),'HOLD')
        a['a']['native_passed']=True;self.assertEqual(gate_status(['a'],a,[]),'PASS')
    def test_maintenance(self):
        self.assertEqual(seconds_allowed(0,20000),10800);self.assertEqual(seconds_allowed(0,1700),0)
    def test_canaries_unique_extremes(self):
        r=[dict(key=str(i),features=feature(30000,c)) for i,c in enumerate((.1,.4,.9))]
        chosen=canaries(r);self.assertEqual(len(set(chosen)),2);self.assertEqual(set(chosen),{'0','2'})

if __name__=='__main__':unittest.main()
