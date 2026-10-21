import json
import shutil
import tempfile
from pathlib import Path
import unittest

from select_resolution import select
from tests import test_benchmark_artifacts as fixtures


class ResolutionSelectionTests(unittest.TestCase):
    def test_tie_rule_and_no_test_prediction_dependency(self):
        fixture=fixtures.BenchmarkArtifactTests(); fixture.setUp()
        try:
            with tempfile.TemporaryDirectory() as temp:
                high=Path(temp)/'high'; shutil.copytree(fixture.root,high)
                config=json.loads((high/'inference_config.json').read_text())
                config['image_size']=320
                (high/'inference_config.json').write_text(json.dumps(config))
                # Removing every test prediction ensures selection cannot depend on them.
                for root in [fixture.root,high]:
                    for path in root.rglob('test_predictions*.csv'): path.unlink()
                result=select(fixture.root,high)
                self.assertEqual(result['selected']['image_size'],224)
                self.assertEqual(result['selected']['architecture'],'densenet121')
                self.assertFalse(result['test_predictions_accessed'])
                manifest=high/'split_manifest.csv'
                manifest.write_text(manifest.read_text().replace('train-0','changed-train-0'))
                with self.assertRaisesRegex(ValueError,'Development cohorts differ'):
                    select(fixture.root,high)
        finally:
            fixture.tearDown()
