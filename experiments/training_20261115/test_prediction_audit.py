"""Check independent score auditing on known decisions and damaged exports."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

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

    def test_cutoff_csv_parser_round_trip_is_traced(self):
        frame = self.frame.copy()
        frame['threshold'] = 0.4289341817299524
        result = audit.check_neural_cutoff_serialization(frame, 0.42893418172995246)
        self.assertEqual(result['difference'], -5.551115123125783e-17)
        self.assertTrue(result['stored_and_fresh_decisions_match_frozen_json'])

    def test_cutoff_boundary_change_is_rejected(self):
        frame = self.frame.copy()
        frozen = 0.42893418172995246
        parsed = float(np.nextafter(frozen, 0))
        frame['threshold'] = parsed
        frame.loc[0, ['probability', 'fresh_cpu_probability']] = parsed
        frame.loc[0, ['prediction', 'fresh_cpu_prediction']] = 1
        with self.assertRaisesRegex(ValueError, 'authoritative frozen JSON'):
            audit.check_neural_cutoff_serialization(frame, frozen)

    def test_unexplained_cutoff_metadata_change_is_rejected(self):
        frame = self.frame.copy()
        frame['threshold'] = 0.4289341817299525
        with self.assertRaisesRegex(ValueError, 'CSV parser path'):
            audit.check_neural_cutoff_serialization(frame, 0.42893418172995246)


if __name__ == '__main__': unittest.main()
