import csv
import json
from pathlib import Path
import tempfile
import unittest
from predict import load_config
from validate_artifacts import validate


class ArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        manifest = []
        for split in ['train', 'validation', 'test']:
            rows = [{'path': f'{split}/{label}.jpg', 'label': str(label),
                     'split': split, 'patient': f'{split}-{label}',
                     'sha256': f'hash-{split}-{label}'} for label in [0, 1]]
            manifest.extend(rows)
            if split != 'train':
                predictions = [dict(r, y_true=r['label'], probability=.1 if r['label']=='0' else .9,
                                    prediction=r['label']) for r in rows]
                self.write_csv(f'{split}_predictions.csv', predictions)
        self.write_csv('split_manifest.csv', manifest)
        self.metrics = {'threshold': .5, 'test': {'accuracy': 1., 'sensitivity': 1., 'specificity': 1.},
                        'validation': {'accuracy': 1., 'sensitivity': 1., 'specificity': 1.}}
        self.write_json('metrics.json', self.metrics)
        for name in ['pneumonia_model.pt', 'best_weights.pt']:
            (self.root/name).write_bytes(b'test placeholder; not a real model')

    def tearDown(self):
        self.temp.cleanup()

    def write_json(self, name, value):
        (self.root/name).write_text(json.dumps(value))

    def write_csv(self, name, rows):
        with (self.root/name).open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_consistent_artifacts(self):
        self.assertTrue(validate(self.root)['validated'])

    def test_rejects_wrong_reported_accuracy(self):
        self.metrics['test']['accuracy'] = .8
        self.write_json('metrics.json', self.metrics)
        with self.assertRaisesRegex(ValueError, 'accuracy'):
            validate(self.root)

    def test_rejects_changed_prediction_identity(self):
        with (self.root/'test_predictions.csv').open() as file:
            rows = list(csv.DictReader(file))
        rows[0]['sha256'] = 'wrong-image'
        self.write_csv('test_predictions.csv', rows)
        with self.assertRaisesRegex(ValueError, 'split manifest'):
            validate(self.root)

    def test_rejects_patient_leakage(self):
        with (self.root/'split_manifest.csv').open() as file:
            rows = list(csv.DictReader(file))
        rows[-1]['patient'] = rows[0]['patient']
        self.write_csv('split_manifest.csv', rows)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate(self.root)

    def test_rejects_missing_checkpoint_and_invalid_threshold(self):
        config = {'architecture': 'efficientnet_b0', 'classes': ['NORMAL', 'PNEUMONIA'],
                  'threshold': .5, 'selected_checkpoints': ['missing.pt']}
        self.write_json('inference_config.json', config)
        with self.assertRaisesRegex(ValueError, 'Checkpoint'):
            load_config(self.root)
        config['threshold'] = float('nan')
        self.write_json('inference_config.json', config)
        with self.assertRaisesRegex(ValueError, 'threshold'):
            load_config(self.root)

    def test_benchmark_four_partitions_and_calibration_leakage(self):
        manifest = []
        for split in ['train', 'selection', 'calibration', 'test']:
            rows = [dict(path=f'{split}/{y}.jpg', label=str(y), split=split,
                         patient=f'{split}-{y}', sha256=f'{split}-{y}') for y in [0, 1]]
            manifest.extend(rows)
            if split != 'train':
                self.write_csv(f'{split}_predictions.csv', [dict(r, y_true=r['label'],
                    probability=.1 if r['label']=='0' else .9, prediction=r['label']) for r in rows])
        self.write_csv('split_manifest.csv', manifest)
        self.write_json('metrics.json', dict(threshold=.5, temperature=2., test=self.metrics['test']))
        self.write_json('inference_config.json', dict(architecture='resnet18',
            classes=['NORMAL', 'PNEUMONIA'], threshold=.5, temperature=2.,
            selected_checkpoints=['best_weights.pt']))
        self.assertTrue(validate(self.root)['validated'])
        manifest[4]['patient'] = manifest[2]['patient']
        self.write_csv('split_manifest.csv', manifest)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate(self.root)

    def test_missing_test_metrics_are_rejected(self):
        del self.metrics['test']
        self.write_json('metrics.json', self.metrics)
        with self.assertRaisesRegex(ValueError, 'Missing reported metrics'):
            validate(self.root)


if __name__ == '__main__':
    unittest.main()
