from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,json,unittest,os,sys,types
import campaign as C

class Fake:
    def usage(self):return 10,20,''
    def allocation(self,j,c,m):return f'{j}:{c}:{m}'
    def find_job(self,n):return None

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=TemporaryDirectory();self.base=Path(self.t.name)
        self.patches=[patch.object(C,'ROOT',self.base/'root'),patch.object(C,'CAMPAIGN',self.base/'campaign'),
                      patch.object(C,'HERE',self.base/'release'),patch.object(C,'S',dict(maintenance_start='2099-01-01T00:00:00'))]
        if os.name=='nt':
            self.patches.append(patch.dict(sys.modules,{'fcntl':types.SimpleNamespace(LOCK_EX=1,LOCK_NB=4,flock=lambda *a:None)}))
        for p in self.patches:p.start()
        C.HERE.mkdir();(C.HERE/'source_lock.json').write_text('{}')
    def tearDown(self):
        for p in reversed(self.patches):p.stop()
        self.t.cleanup()
    def test_hash_portable(self):
        f=self.base/'a';b=b'abc'*500000;f.write_bytes(b)
        self.assertEqual(C.sha(f),hashlib.sha256(b).hexdigest())
    def test_write_exclusive(self):
        p=C.ROOT/'x.json';C.write(p,dict(a=1))
        with self.assertRaises(FileExistsError):C.write(p,dict(a=2))
        self.assertEqual(C.read(p),dict(a=1))
    def test_write_outside_rejected(self):
        with self.assertRaises(AssertionError):C.write(self.base/'outside.json',{})
    def fixture(self):
        out=C.CAMPAIGN/'out';(out/'prepared').mkdir(parents=True)
        for n in ('raw_unplanned.bin','geometry_unplanned.bin'):(out/'prepared'/n).write_bytes(n.encode())
        a=dict(window='Win1',slice_index='1',exact_replay_seed='123',selected_sample_index='45')
        original=dict(case_key='test',slot=1,assignment=a)
        rp=C.CAMPAIGN/'wave01/accepted/test/core001.json';rp.parent.mkdir(parents=True)
        d=dict(complete=True,memory={'passed':True},source_lock_sha256=C.LOCK,record_sha256=C.digest(original),
               case='test',slot=1,output=str(out),preparation=dict(seed=123,sample=45,window='Win1'),
               features=dict(num_cells=2),files={'prepared/'+n:C.sha(out/'prepared'/n) for n in ('raw_unplanned.bin','geometry_unplanned.bin')})
        rp.write_text(json.dumps(d));return C.record(original,rp,d),rp
    def test_replay_identity(self):
        r,_=self.fixture();C.validate_replay(r,True);self.assertEqual(r['region'],9)
    def test_wrong_seed(self):
        r,_=self.fixture();r['seed']=124
        with self.assertRaises(AssertionError):C.validate_replay(r,True)
    def test_wrong_case(self):
        r,_=self.fixture();r['case']='other'
        with self.assertRaises(AssertionError):C.validate_replay(r,True)
    def test_input_changed(self):
        r,_=self.fixture();Path(r['raw']).write_bytes(b'changed')
        with self.assertRaises(AssertionError):C.validate_replay(r,True)
    def test_receipt_changed(self):
        r,rp=self.fixture();rp.write_text('{}')
        with self.assertRaises(AssertionError):C.validate_replay(r)
    def test_no_duplicate_real_submit_path(self):
        with patch.object(C,'library',return_value=Fake()),patch.object(C.subprocess,'check_output',return_value='123\n') as sb:
            self.assertEqual(C.submit('worker','test',2,4,3600,'core'),'123')
            self.assertEqual(C.submit('worker','test',2,4,3600,'core'),'123')
            self.assertEqual(sb.call_count,1)
            args=sb.call_args.args[0]
            self.assertIn('--account=mit_amf_advanced_cpu',args);self.assertIn('--qos=mit_amf_advanced_cpu',args)
            self.assertIn('--exclude=node1622',args)
    def test_lost_ack_not_repeated(self):
        # A submission intent with no recoverable job must never cause a second sbatch.
        p=C.claim_path('test');C.write(p,dict(name='w60n1_'+C.HERE.name[:6]+'_test',source_lock=C.sha(C.HERE/'source_lock.json')))
        with patch.object(C,'library',return_value=Fake()),patch.object(C.subprocess,'check_output') as sb:
            with self.assertRaises(AssertionError):C.submit('worker','test',2,4,3600)
            sb.assert_not_called()

if __name__=='__main__':unittest.main()
