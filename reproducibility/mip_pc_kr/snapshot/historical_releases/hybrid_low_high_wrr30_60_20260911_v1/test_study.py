import unittest
import numpy as np
from study import medoid, combine_table, toml_values, reference_deck
import tomllib


class HybridTests(unittest.TestCase):
    def test_medoid_tie_by_original_source_row(self):
        g=[dict(EffectiveSwi=.2,ReplaySourceRow=i+1,SliceIndex=87-i) for i in range(87)]
        self.assertEqual(medoid(g)['ReplaySourceRow'],1)

    def test_median_not_mean(self):
        g=[dict(EffectiveSwi=i/100,ReplaySourceRow=i+1,SliceIndex=i+1) for i in range(87)]
        self.assertEqual(medoid(g)['ReplaySourceRow'],44)

    def test_union_preserves_pc_and_kr(self):
        p=np.array([[0,0,1,1e4],[.01,.02,.98,1e4],[.5,.5,.2,1e5],[.8,1,0,1e7]])
        k=np.array([[0,0,1],[.1,.05,.8],[.4,.3,.3],[.8,1,0]])
        t,e=combine_table(p,k)
        x=np.linspace(0,.8,10001)
        np.testing.assert_allclose(np.interp(x,t[:,0],t[:,3]),np.interp(x,p[:,0],p[:,3]),rtol=2e-15,atol=1e-8)
        for j in (1,2):
            np.testing.assert_allclose(np.interp(x,t[:,0],t[:,j]),np.interp(x,k[:,0],k[:,j]),atol=1e-15)

    def test_wrong_endpoint_rejected(self):
        p=np.array([[0,0,1,1],[.8,1,0,2]])
        k=np.array([[0,0,1],[.9,1,0]])
        with self.assertRaises(AssertionError):combine_table(p,k)

    def test_nonmonotone_rejected(self):
        p=np.array([[0,0,1,1],[.8,1,0,2]])
        k=np.array([[0,0,1],[.4,.9,.5],[.8,.2,0]])
        with self.assertRaises(AssertionError):combine_table(p,k)

    def test_toml_roundtrip(self):
        d=dict(retain_years=[.001,25.,50.,100.,1000.],production_promotion=False,name='hybrid',steps=210)
        self.assertEqual(tomllib.loads(toml_values(d)),d)

    def test_clay_only_deck_and_pressure_units(self):
        sg30=np.r_[0,.001,np.linspace(0,.7,30)[1:-1],.699,.7]
        sg60=np.r_[0,.001,np.linspace(0,.8,30)[1:-1],.799,.8]
        pc30=2.8*np.exp(4.2*sg30);pc30[0]=0
        old=np.column_stack([sg30,(sg30/.7)**1.7,(1-sg30/.7)**4.8,pc30])
        swstar=1-sg60/.8
        clay=np.column_stack([sg60,(1-swstar)**2.5,swstar**3.5,
                              1e5*2.8*np.cos(np.pi/3)/np.cos(np.pi/6)*np.exp(4.2*sg60)])
        clay[0,3]=0
        prefix='METRIC\nSGOF\n0 0 1 0\n.95 1 0 11\n/'
        suffix='/\nDENSITY\n1000 1000 600 /\n'
        text=prefix+'\n'+'\n'.join(' '.join(map(str,row)) for row in old)+'\n'+suffix
        changed=reference_deck(text,clay)
        self.assertTrue(changed.startswith(prefix));self.assertTrue(changed.endswith(suffix))
        parsed=np.loadtxt(changed[len(prefix):-len(suffix)].splitlines())
        np.testing.assert_allclose(parsed[:,:3],clay[:,:3],rtol=0,atol=0)
        np.testing.assert_allclose(parsed[:,3]*1e5,clay[:,3],rtol=2e-16)

    def test_wrong_clay_wettability_rejected(self):
        with self.assertRaises(AssertionError):
            reference_deck('SGOF\n0 0 1 0 /\n0 0 1 0 /',np.zeros((32,4)))


if __name__=='__main__':unittest.main()
