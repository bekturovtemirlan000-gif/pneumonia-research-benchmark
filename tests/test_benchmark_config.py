import json
from pathlib import Path
import tempfile
import unittest
from predict import load_config


class BenchmarkConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'trial').mkdir()
        (self.root / 'trial' / 'best_weights.pt').write_bytes(b'placeholder')
        self.config = dict(architecture='densenet121', classes=['NORMAL', 'PNEUMONIA'],
                           threshold=.7, temperature=2.,
                           selected_checkpoints=['trial/best_weights.pt'])

    def tearDown(self):
        self.temp.cleanup()

    def load(self):
        (self.root / 'inference_config.json').write_text(json.dumps(self.config))
        return load_config(self.root)

    def test_each_architecture_accepts_nested_checkpoint(self):
        for architecture in ['efficientnet_b0', 'densenet121', 'resnet18']:
            self.config['architecture'] = architecture
            _, paths = self.load()
            self.assertEqual(paths, [(self.root / 'trial' / 'best_weights.pt').resolve()])

    def test_rejects_outside_existing_checkpoint(self):
        with tempfile.NamedTemporaryFile(suffix='.pt') as file:
            self.config['selected_checkpoints'] = [file.name]
            with self.assertRaisesRegex(ValueError, 'inside'):
                self.load()

    def test_rejects_invalid_temperatures(self):
        for temperature in [0, -1, float('nan'), float('inf'), '2']:
            self.config['temperature'] = temperature
            with self.assertRaisesRegex(ValueError, 'temperature'):
                self.load()

    def test_resolution_must_be_supported(self):
        self.config['image_size'] = 320
        self.assertEqual(self.load()[0]['image_size'], 320)
        self.config['image_size'] = 999
        with self.assertRaisesRegex(ValueError, 'image size'):
            self.load()


if __name__ == '__main__':
    unittest.main()
