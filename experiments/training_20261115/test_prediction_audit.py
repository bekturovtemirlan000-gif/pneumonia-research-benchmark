"""Check independent score auditing on known decisions and damaged exports."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import audit_prediction_tables as audit


class PredictionAuditTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(dict(sha256=[str(i)*64 for i in range(4)],
            patient=['a', 'b', 'c', 'd'], y_true=[0, 0, 1, 1],
            probability=[.1, .7, .3, .9], prediction=[0, 1, 0, 1], threshold=[.5]*4,
            fresh_cpu_probability=[.1, .7, .3, .9], fresh_cpu_prediction=[0, 1, 0, 1]))

    def test_known_confusion_and_brier(self):
        result = audit.metrics(self.frame)
        self.assertEqual([result[k] for k in ['tn', 'fp', 'fn', 'tp']], [1, 1, 1, 1])
        self.assertEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['brier_score'], .25)
        self.assertAlmostEqual(result['auroc'], .75)

    def test_strict_cpu_failure_is_preserved(self):
        frame = self.frame.copy()
        frame.loc[0, 'fresh_cpu_probability'] = .500001
        frame.loc[0, 'fresh_cpu_prediction'] = 1
        result = audit.check_cpu(frame, dict(max_absolute_score_difference=.400001,
                                            changed_predictions=1, passed=False), 'fixture')
        self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'pass/fail'):
            audit.check_cpu(frame, dict(max_absolute_score_difference=.400001,
                                       changed_predictions=1, passed=True), 'fixture')

    def test_csv_decisions_checked_after_byte_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            original = audit.HERE
            try:
                audit.HERE = Path(folder)
                (audit.HERE/'predictions').mkdir()
                path = audit.HERE/'predictions/public_test_predictions.csv'
                damaged = self.frame.copy()
                damaged.loc[0, 'prediction'] = 1
                path.write_bytes(damaged.to_csv(index=False, lineterminator='\n', float_format='%.17g').encode())
                meta = dict(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            columns=list(damaged.columns), rows=4)
                with self.assertRaisesRegex(ValueError, 'frozen cutoffs'):
                    audit.table(meta, 4, 4)
            finally:
                audit.HERE = original


if __name__ == '__main__': unittest.main()
