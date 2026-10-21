"""Regression contracts for deployment boundaries and complete CPU evidence."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from experiments.wshy.online_memory_2b.isolation import configure, contained, output_path, write_root
from experiments.wshy.online_memory_2b.protocol import save, sha
from experiments.wshy.online_memory_2b.receipts import (INHERITED_COUNTS, WINDOWS_OWNER_TEST,
    WINDOWS_OWNER_REASON, inherited_result, successful_counts, verify_inherited)
from experiments.wshy.online_memory_2b.runner import isolated_readout


def successful_log(counts=INHERITED_COUNTS):
    return ''.join(f'Ran {count} tests in 0.100s\n\nOK\n' for count in counts)


class DeploymentContracts(unittest.TestCase):
    def test_output_refuses_parent_escape_before_creating_any_file(self):
        outside = write_root().parent / 'forbidden-online-memory-output.json'
        with self.assertRaisesRegex(ValueError, 'write boundary'):
            save(outside, {'must_not_write': True})

    def test_containment_rejects_root_and_relative_parent_escape(self):
        root = write_root()
        for path in (root, root / '..' / 'outside'):
            with self.assertRaises(ValueError):
                contained(path, root)
        self.assertEqual(contained(root/'runs'/'new-run', root), root/'runs'/'new-run')

    @unittest.skipIf(os.name == 'nt', 'cluster contract uses POSIX symlink creation')
    def test_existing_symlink_cannot_redirect_output_outside_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / 'escape'
            link.symlink_to(write_root().parent, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'write boundary'):
                output_path(link/'must-not-create')

    def test_cache_configuration_overrides_external_defaults(self):
        old_tempdir = tempfile.tempdir
        try:
            with patch.dict(os.environ, {'TMPDIR':'/outside', 'HF_HOME':'/outside',
                                         'XDG_CACHE_HOME':'/outside', 'TRANSFORMERS_CACHE':'/outside'}):
                root = configure()
                for key in ('TMPDIR','TEMP','TMP','HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME',
                            'TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR',
                            'PYTORCH_KERNEL_CACHE_PATH','TRITON_CACHE_DIR','CUDA_CACHE_PATH',
                            'XDG_CONFIG_HOME','XDG_DATA_HOME','XDG_STATE_HOME'):
                    self.assertTrue(Path(os.environ[key]).resolve().is_relative_to(root))
                self.assertNotIn('TRANSFORMERS_CACHE', os.environ)
                self.assertTrue(Path(tempfile.gettempdir()).is_relative_to(root))
        finally:
            tempfile.tempdir = old_tempdir


class RegressionReceiptContracts(unittest.TestCase):
    def test_all_four_successful_suites_are_required(self):
        self.assertEqual(successful_counts(successful_log(), INHERITED_COUNTS), INHERITED_COUNTS)
        for log in (successful_log([3,12,185,8]), successful_log([3,12,186]),
                    successful_log().replace('OK', 'OK (skipped=1)', 1),
                    successful_log().replace('OK', 'FAILED (errors=1)', 1)):
            with self.assertRaises(ValueError):
                successful_counts(log, INHERITED_COUNTS)
        line = (WINDOWS_OWNER_TEST.rsplit('.',1)[1]+' ('+WINDOWS_OWNER_TEST+
                ") ... skipped '"+WINDOWS_OWNER_REASON+"'\n")
        known = ''.join((line if i==2 else '')+f'Ran {n} tests in 0.100s\n\n'+
                        ('OK (skipped=1)' if i==2 else 'OK')+'\n'
                        for i,n in enumerate(INHERITED_COUNTS))
        result = inherited_result(known, 'linux')
        self.assertEqual((result['tests_discovered'],result['tests_run'],result['tests_skipped']), (209,208,1))
        for bad in (known.replace(WINDOWS_OWNER_TEST, 'other.test'),
                    known.replace(WINDOWS_OWNER_REASON,'other reason'),
                    known.replace('skipped=1','skipped=2'),
                    known.replace(line,line+line),known.replace('Ran 186','Ran 185')):
            with self.assertRaises(ValueError): inherited_result(bad,'linux')
        with self.assertRaises(ValueError): inherited_result(known,'win32')
        self.assertEqual(inherited_result(successful_log())['tests_run'],209)

    def fixture(self, directory):
        out = Path(directory)
        (out/'inherited.txt').write_text(successful_log(), encoding='utf-8')
        save(out/'inherited.json', dict(status='passed', **inherited_result(successful_log()),
             source_lock_sha256='source-a', log_sha256=sha(out/'inherited.txt')))
        return out, sha(out/'inherited.json'), sha(out/'inherited.txt')

    def test_receipt_cannot_be_reused_after_source_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            out, receipt_hash, log_hash = self.fixture(directory)
            self.assertEqual(verify_inherited(out, 'source-a', receipt_hash, log_hash)['tests_run'], 209)
            with self.assertRaisesRegex(ValueError, '209-test receipt'):
                verify_inherited(out, 'source-b', receipt_hash, log_hash)

    def test_modified_log_cannot_reuse_a_successful_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            out, receipt_hash, log_hash = self.fixture(directory)
            (out/'inherited.txt').write_text(successful_log([3,12,185,8]), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                verify_inherited(out, 'source-a', receipt_hash, log_hash)

    def test_receipt_totals_are_checked_even_if_its_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            out, _, log_hash = self.fixture(directory)
            receipt = json.loads((out/'inherited.json').read_text(encoding='utf-8'))
            receipt['tests_run'] = 208
            (out/'inherited.json').write_text(json.dumps(receipt), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '209-test receipt'):
                verify_inherited(out, 'source-a', sha(out/'inherited.json'), log_hash)


class ReadoutStateContracts(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNamespace(model=SimpleNamespace(rope_deltas=torch.tensor([3])))
        state = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, state)

    def test_fresh_readout_can_leave_rng_and_position_unchanged(self):
        env = dict(torch=torch, answer=lambda *args: dict(text='A'))
        self.assertEqual(isolated_readout(env, self.model, None, [], 'q'), dict(text='A'))

    def test_readout_rng_mutation_is_rejected(self):
        def mutate(*args):
            torch.rand(1)
            return dict(text='A')
        with self.assertRaisesRegex(RuntimeError, 'RNG state'):
            isolated_readout(dict(torch=torch, answer=mutate), self.model, None, [], 'q')

    def test_readout_position_mutation_is_rejected(self):
        def mutate(*args):
            self.model.model.rope_deltas.add_(1)
            return dict(text='A')
        with self.assertRaisesRegex(RuntimeError, 'position state'):
            isolated_readout(dict(torch=torch, answer=mutate), self.model, None, [], 'q')
