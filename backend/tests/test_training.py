"""Tests practice training and the fixed model's displayed evidence."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from backend.training_service import TrainingService
from training.final_model import ResidualMLP, fit_practice, verify_freeze


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path('training/results/three_class_residual_f1_v1')
        metadata = json.loads((self.folder / 'run.json').read_text())
        self.service = TrainingService('data', self.temp.name, self.folder, metadata)
        self.addCleanup(self.service.close)

    def test_metrics_distinguish_seed_42_from_exploration(self):
        metrics = self.service.metrics
        self.assertEqual(metrics['class_names'], ['Hepatocyte', 'Erythroblast', 'Proerythroblast'])
        self.assertEqual(metrics['splits']['test']['n_errors'], 782)
        self.assertEqual(metrics['baseline']['test']['n_errors'], 802)
        self.assertTrue(metrics['improved_has_lower_error'])
        self.assertEqual(metrics['validation_exploration']['seeds'], [42, 43, 44])
        self.assertAlmostEqual(metrics['validation_exploration']['improved_macro_f1_mean'],
                               0.486815, places=5)
        self.assertEqual(metrics['history'][0]['epoch'], 0)
        self.assertIn('used by earlier models', metrics['evaluation_note'])

    def test_frozen_artifacts_and_implementation_match(self):
        record = verify_freeze(self.folder)
        self.assertEqual(record['test'], 'replicate 4, previously used by older models')
        self.assertLess(json.loads((self.folder / 'freeze.json').read_text())['frozen_utc'],
                        json.loads((self.folder / 'test_access_started.json').read_text())['started_utc'])

    def test_practice_fits_f1_checkpoint_without_changing_frozen_weights(self):
        # Synthetic inputs exercise real practice training without opening biological data.
        x = torch.rand(6, 1176)
        y = torch.tensor([0, 1, 2, 0, 1, 2])
        schema = {key: self.service.metadata[key] for key in
                  ['cleaning', 'feature_names', 'class_names', 'filter_counts', 'loci']}
        frozen = hashlib.sha256((self.folder / 'weights.pt').read_bytes()).hexdigest()
        with patch.object(self.service, '_prepare_inputs', return_value=((x, y, x, y), schema)):
            job = self.service.start(2, dropout=0.3)
            self.service.executor.submit(lambda: None).result(timeout=30)
        result = self.service.snapshot(job['id'])
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(len(result['history']), 6)
        self.assertEqual(result['stopping']['comparison'],
                         'maximum_macro_f1_then_minimum_loss_on_ties')
        saved = json.loads((Path(result['result_folder']) / 'run.json').read_text())
        self.assertFalse(saved['test_evaluated'])
        self.assertEqual(saved['dropout'], 0.3)
        self.assertEqual(frozen, hashlib.sha256((self.folder / 'weights.pt').read_bytes()).hexdigest())

    def test_practice_schema_and_source_mismatch_are_rejected(self):
        self.service.metadata['source_sha256'] = {'input.csv': 'incorrect'}
        self.service.data_directory = Path(self.temp.name)
        (Path(self.temp.name) / 'input.csv').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Training source changed'):
            self.service._prepare_inputs()


if __name__ == '__main__':
    unittest.main()
