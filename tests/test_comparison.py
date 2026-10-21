import tempfile
import unittest
from pathlib import Path

import pandas as pd

from compare_experiments import compare


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a = Path(self.temp.name) / 'a'
        self.b = Path(self.temp.name) / 'b'
        self.a.mkdir()
        self.b.mkdir()
        self.frame = pd.DataFrame({
            'sha256': ['a', 'b', 'c', 'd'], 'patient': ['p1', 'p1', 'p2', 'p2'],
            'y_true': [0, 1, 0, 1], 'probability': [.1, .8, .2, .9],
            'prediction': [0, 1, 0, 1]})
        self.frame.to_csv(self.a / 'test_predictions.csv', index=False)

    def test_row_order_does_not_change_pairing(self):
        self.frame.iloc[::-1].to_csv(self.b / 'test_predictions.csv', index=False)
        result = compare(self.a, self.b, replicates=10)
        self.assertTrue(all(v == 0 for v in result['difference'].values()))
        self.assertTrue(all(v == [0., 0.] for v in result['paired_cluster_bootstrap_95ci'].values()))

    def test_changed_cohort_is_rejected(self):
        self.frame.loc[0, 'sha256'] = 'different_image'
        self.frame.to_csv(self.b / 'test_predictions.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'cohorts'):
            compare(self.a, self.b, replicates=10)

    def test_frozen_prediction_change_is_measured(self):
        self.frame.loc[0, 'prediction'] = 1
        self.frame.to_csv(self.b / 'test_predictions.csv', index=False)
        result = compare(self.a, self.b, replicates=10)
        self.assertEqual(result['difference']['specificity'], -.5)
        self.assertEqual(result['difference']['auroc'], 0.)


if __name__ == '__main__':
    unittest.main()
