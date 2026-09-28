import importlib.util
from pathlib import Path
import unittest

import resource
import remote


class PilotTests(unittest.TestCase):
    def test_fractional_slurm_memory_is_exact(self):
        template='Account=mit_amf_advanced_cpu QOS=mit_amf_advanced_cpu Partition=mit_normal NumCPUs=2 NumNodes=1 MinMemoryNode='
        for value, mib in [('4.50G',4608),('2.75G',2816),('6.50G',6656),('768M',768),('3G',3072)]:
            resource.parse_allocation(template+value,2,mib)
        for value in ('4.49G','4.51G','4608.01M','4.50T','NaN'):
            with self.assertRaises(AssertionError):
                resource.parse_allocation(template+value,2,4608)

    def test_allocation_requires_advanced_and_exact_memory(self):
        text='Account=mit_amf_advanced_cpu QOS=mit_amf_advanced_cpu Partition=mit_normal NumCPUs=2 NumNodes=1-1 MinMemoryNode=2816M'
        resource.parse_allocation(text,2,2816)
        for old,new in [('QOS=mit_amf_advanced_cpu','QOS=normal'),('2816M','3G'),('NumCPUs=2','NumCPUs=8')]:
            with self.assertRaises(AssertionError):resource.parse_allocation(text.replace(old,new),2,2816)

    def test_whole_job_memory_uses_relative_gate(self):
        d=dict(peak_bytes=2*1024**3,limit_bytes=2816*1024**2,events=0)
        self.assertTrue(resource.passed(d,2816))
        self.assertFalse(resource.passed(dict(d,events=1),2816))
        self.assertFalse(resource.passed(dict(d,peak_bytes=2750*1024**2),2816))

    def test_stage_graph_preserves_prefix(self):
        jobs={k:dict(job=str(i)) for i,k in enumerate(('bootstrap','small','larger','prefix','resume','collect'),1)}
        self.assertEqual(remote.stage_dependencies('resume',jobs),['4'])
        self.assertEqual(remote.stage_dependencies('collect',jobs),['2','3','5'])
        self.assertEqual(remote.stage_dependencies('native',jobs),['6'])

    def test_extracted_kernel_is_verbatim(self):
        here=Path(__file__).resolve().parent
        old=(here.parents[1]/'kr_sparse10_validation_20260916_v1/benchmark.jl').read_text()
        expected=old[old.index('function writecsv('):old.index('\nfunction main()')]
        self.assertEqual((here/'Kernel.jl').read_text(),expected)
        worker=(here/'worker.jl').read_text()
        self.assertIn('run_curve(input,g,out,"sparse10")',worker)
        self.assertNotIn('run_curve(input,g,out,"full100")',worker)


if __name__=='__main__':unittest.main()
