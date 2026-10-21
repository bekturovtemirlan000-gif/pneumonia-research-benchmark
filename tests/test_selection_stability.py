import json
from pathlib import Path
import shutil
import tempfile
import unittest

from inspect_selection_stability import inspect
from tests import test_benchmark_artifacts as fixtures


class SelectionStabilityTests(unittest.TestCase):
    def test_tied_models_without_any_test_predictions(self):
        fixture=fixtures.BenchmarkArtifactTests();fixture.setUp()
        try:
            with tempfile.TemporaryDirectory() as temp:
                high=Path(temp)/'high';shutil.copytree(fixture.root,high)
                config=json.loads((high/'inference_config.json').read_text())
                config['image_size']=320
                (high/'inference_config.json').write_text(json.dumps(config))
                for root in [fixture.root,high]:
                    for path in root.rglob('test_predictions*.csv'): path.unlink()
                report=inspect(fixture.root,high,replicates=40)
                self.assertGreater(report['usable_replicates'],0)
                self.assertEqual(report['winner_fractions']['224:densenet121'],1.)
                self.assertEqual(sum(report['winner_counts'].values()),report['usable_replicates'])
                self.assertEqual(report['replicates_with_tied_maximum'],report['usable_replicates'])
                self.assertFalse(report['test_predictions_accessed'])
                self.assertFalse(report['changes_to_deployed_choice'])
                self.assertEqual(len(report['winner_counts']),6)
        finally:
            fixture.tearDown()
