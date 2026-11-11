import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import common as C
import recovery as R


class RecoveryTests(unittest.TestCase):
    def test_archive_wins_over_missing_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'accepted.zip').touch()
            self.assertEqual(R.original_state(p/'missing', p/'accepted.zip'), 'archive_excluded')

    def test_saved_outputs_without_receipt_never_become_unstarted(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'flow').mkdir(); (p/'flow/pc100.csv').touch()
            self.assertEqual(R.original_state(p, p/'absent'), 'hold_unreceipted_output')

    def test_interrupted_requires_started_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'SOLVE_STARTED.toml').touch()
            self.assertEqual(R.original_state(p, p/'absent'), 'interrupted')
            (p/'FLOW_ERROR.toml').touch()
            self.assertEqual(R.original_state(p, p/'absent'), 'hold_error')

    def test_complete_saved_flow_has_priority_over_old_postflow_hold(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'flow').mkdir(); (p/'flow/CORE_PASS.toml').touch(); (p/'POSTFLOW_HOLD.json').touch()
            self.assertEqual(R.original_state(p, p/'absent'), 'saved_flow')

    def test_saved_copy_is_checked_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); (p/'a').write_bytes(b'original'); (p/'b').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                R.copied_file(p/'a', p/'b')
            self.assertEqual((p/'b').read_bytes(), b'changed')

    def test_packet_shapes(self):
        records = [dict(slot=i) for i in range(1, 34)]
        rows = [dict(slot=i, category='interrupted' if i < 4 else 'unstarted') for i in range(1, 34)]
        numerical = R.packetize(rows, records, 'solve')
        self.assertEqual([len(x) for x in numerical], [1, 1, 1, 8, 8, 8, 6])
        self.assertEqual([len(x) for x in R.packetize(rows, records, 'finish')], [32, 1])
        self.assertEqual(sorted(r['slot'] for p in numerical for r in p), list(range(1, 34)))

    def test_immutable_claim_cannot_be_renamed_between_phases(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'claim.json'
            C.write(p, dict(phase='pilot'))
            with self.assertRaisesRegex(RuntimeError, 'Immutable'):
                C.write(p, dict(phase='solve'))

    def test_wrong_output_root_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Wrong recovery output'):
            R.authorize_recovery(dict(purpose='round2', phase='finish'))

    def test_unknown_phase_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid recovery phase'):
            R.authorize_recovery(dict(purpose='round2', phase='retry_everything'))

    def test_stage_context_restores_root_on_exception(self):
        root = C.ROOT
        with self.assertRaises(ValueError):
            with R.at_root(Path('different')):
                raise ValueError('fixture')
        self.assertEqual(C.ROOT, root)

    def test_launcher_uses_fixed_source_not_slurm_spool(self):
        text = (C.CODE/'job.sh').read_text()
        self.assertIn('CODE=/orcd/data/juanes/001/shaowen/phase1_round2_20261109_v4', text)
        self.assertNotIn('dirname', text)

    def test_exact_input_guard_precedes_new_solve(self):
        text = (C.CODE/'flow_packet.jl').read_text()
        self.assertLess(text.index('TOML.parsefile(prior)==proof'), text.index('SOLVE_STARTED.toml'))
        self.assertLess(text.index('TOML.parsefile(prior)==proof'), text.index('pc,kr=run_cached_adaptive'))


def qualify_round2_finish():
    """Actual round-two worker target/copy/publish path on private saved copies."""
    import hashlib
    import shutil
    import zipfile
    import worker
    root = C.CODE/'qualification/round2_fixture'
    prod = root/'production'; home = prod/'round2'; phase = home/'finish'
    case = C.read(C.CODE/'STAGED.json')['pilot_cases'][0]
    original = C.read(C.CODE/'plans'/f'{case}.json'); records = original['packets'][0]
    seed = R.BASE/'phase1_pc40_adaptive10_runner_20261030_v4/numerical/results'/case
    C.write(prod/'plans'/f'{case}.json', original)
    entries = {}; original_hashes = {}
    for index, record in enumerate(records):
        source = seed/f"core{record['slot']:03d}.zip"
        C.verify_archive(source, record, C.read(C.CODE/'LOCK.json')['method_sha256'])
        key = C.key(record); old = prod/'runs'/case/'cores'/key
        entry = dict(category='saved_flow', record_sha256=C.digest(record), slot=record['slot'], key=key)
        if index == 0:
            dest = prod/'results'/case/source.name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
        else:
            with zipfile.ZipFile(source) as z:
                for name in z.namelist():
                    if name.startswith(('input/', 'flow/')):
                        dest = old/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(z.read(name))
                        original_hashes[str(dest)] = C.sha(dest)
            entry['flow_proof_sha256'] = C.sha(old/'flow/CORE_PASS.toml')
        entries[key] = entry
    plan = dict(purpose='round2', phase='finish', packets=[records], entries=entries,
        original_plan_sha256=C.sha(prod/'plans'/f'{case}.json'), original_job='private_fixture')
    dest = phase/'plans'/f'{case}.json'; C.write(dest, plan)
    C.write(dest.with_suffix('.seal.json'), dict(sha256=C.sha(dest), source=C.sha(C.CODE/'LOCK.json')))
    C.write(phase/'SCOPE.json', dict(purpose='round2', phase='finish', cases=[case], source=C.sha(C.CODE/'LOCK.json')))
    stages = []; process = worker.process
    def finalization_only(args, log, fd, env=None, cwd=None, timeout=6900):
        C.require(log.name in ('runtime.log', 'export.log', 'native.log'), 'Qualification attempted a numerical solve')
        stages.append(log.name)
        return process(args, log, fd, env, cwd, timeout)
    def isolated_authorization(scope):
        C.require(C.ROOT == phase and phase.is_relative_to(C.CODE/'qualification'), 'Fixture escaped isolation')
    with patch.object(R, 'PRODUCTION', prod), patch.object(R, 'HOME', home), patch.object(C, 'ROOT', phase), \
         patch.object(R, 'authorize_recovery', isolated_authorization), patch.object(worker, 'process', finalization_only):
        worker.run(case, 0, finish_only=True)
        for record in records:
            C.verify_archive(prod/'results'/case/f"core{record['slot']:03d}.zip", record, C.read(C.CODE/'LOCK.json')['method_sha256'])
        before = len(stages)
        worker.run(case, 0, finish_only=True)
        C.require(len(stages) == before, 'Accepted round-two no-op launched a stage')
    for name, digest in original_hashes.items():
        C.require(C.sha(Path(name)) == digest, 'Original saved input/flow changed')
    C.write(C.CODE/'qualification/ROUND2_FINISH_LIFECYCLE_PASS.json', dict(passed=True,
        source=C.sha(C.CODE/'LOCK.json'), saved_flows_finalized=31, accepted_excluded=1,
        new_solves=0, originals_preserved=True, stages=stages, production_modified=False))


if __name__ == '__main__':
    unittest.main()
