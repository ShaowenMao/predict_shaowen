import copy
from pathlib import Path
import unittest
import common as C
from cache_contract import verify_cache_evidence,LEGACY_RUNNER

class CacheContract(unittest.TestCase):
    def setUp(self):
        self.science={'DrainageCache.jl':'a','CachedAdaptiveCurve.jl':'b'}
        self.source={'runner_source':'new'}
        self.summary=dict(state_access='verified_per_core_cache_v1',cached_states_verified=100,
             cached_queries_verified=10,kr_states=10,replay_states_checked=0,reference_states_checked=0,
             cells=160000,cache_payload_bytes=416001600)
        self.receipt={'state_cache_implementation':dict(self.science)}
    def check(self,s=None,p=None,source=None):
        return verify_cache_evidence(self.summary if s is None else s,self.receipt if p is None else p,
                                     self.source if source is None else source,'new',self.science)
    def test_cached_receipt(self):self.assertEqual(self.check(),'verified_cache')
    def test_each_guard(self):
        for k,v in [('state_access','unknown'),('cached_states_verified',99),('cached_queries_verified',9),
                    ('replay_states_checked',1),('reference_states_checked',100),('cache_payload_bytes',1)]:
            with self.subTest(k=k),self.assertRaises(RuntimeError):self.check(dict(self.summary,**{k:v}))
    def test_wrong_implementation(self):
        with self.assertRaises(RuntimeError):self.check(p={'state_cache_implementation':{'DrainageCache.jl':'changed'}})
    def test_wrong_source(self):
        with self.assertRaises(RuntimeError):self.check(source={'runner_source':'unknown'})
    def test_legacy_saved_flow_remains_recoverable(self):
        self.assertEqual(self.check(s={},p={},source={'runner_source':LEGACY_RUNNER}),'legacy_saved_flow')
    def test_new_source_cannot_omit_cache_evidence(self):
        with self.assertRaises(RuntimeError):self.check(s={},p={})
    def test_mixed_proof_rejected(self):
        with self.assertRaises(RuntimeError):self.check(s={},source={'runner_source':LEGACY_RUNNER})
    def test_memory_ceiling(self):
        with self.assertRaises(RuntimeError):self.check(dict(self.summary,cells=300000,cache_payload_bytes=780001600))
    def test_fixture_is_not_full_case(self):
        lock=C.read(C.CODE/'LOCK.json');self.assertFalse(lock['full_campaign_authorized'])
        case=C.read(C.CODE/'STAGED.json')['pilot_cases'][0];plan=C.read(C.CODE/'plans'/f'{case}.json')
        self.assertEqual(plan['purpose'],'qualification');self.assertEqual(plan['allowed_packets'],[0])
        self.assertEqual(len(plan['packets'][0]),32);self.assertEqual(len(plan['records']),522)
        records={C.key(r):r for r in plan['records']}
        for r in plan['packets'][0]:self.assertEqual(records[C.key(r)],r)
    def test_scheduler_and_science_checks_unchanged(self):
        old=C.read(C.CODE/'BASELINE_LOCK.json');new=C.read(C.CODE/'LOCK.json')
        for n in ('common.py','planning.py','control.py','slurm.py','prepare_packet.m','export_packet.py','native_packet.jl','runtime_snapshot.py'):
            self.assertEqual(C.sha(C.CODE/n),old['code'][n],n)
        for n,h in old['science'].items():self.assertEqual(new['science'][n],h,n)
    def test_no_reference_replay_in_packet(self):
        s=(C.CODE/'flow_packet.jl').read_text()
        self.assertIn('run_cached_adaptive(x,g,output,identity)',s)
        self.assertNotIn('qualification_reference=true',s)
        self.assertNotIn('pc,kr=run_adaptive(',s)

if __name__=='__main__':unittest.main()
