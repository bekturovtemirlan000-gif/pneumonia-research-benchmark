import unittest
import numpy as np
import pandas as pd
from selection import choose, internal_split, threshold


class SelectionTests(unittest.TestCase):
    def test_known_best_midpoint(self):
        self.assertEqual(threshold([0, 0, 1, 1], [.1, .4, .6, .9]), (.5, 4))

    def test_tie_preserves_sensitivity(self):
        self.assertEqual(threshold([0, 1], [.5, .5]), (0., 1))

    def test_invalid_scores(self):
        for values in [[.2, np.nan], [.2, 1.1], [.2]]:
            with self.assertRaises(ValueError): threshold([0, 1], values)

    def test_internal_split_deterministic_and_grouped(self):
        frame = pd.DataFrame([dict(patient=f'p{g}', label=g % 2, sha256=f'h{g}_{i}') for g in range(40) for i in range(2)])
        fit, stop = internal_split(frame)
        fit2, stop2 = internal_split(frame.sample(frac=1, random_state=7))
        self.assertEqual(len(fit) + len(stop), len(frame))
        self.assertFalse(set(fit.patient) & set(stop.patient))
        self.assertEqual(set(stop.sha256), set(stop2.sha256))
        self.assertEqual(set(fit.sha256), set(fit2.sha256))

    def test_model_choice_is_grouped_and_tie_prefers_smaller(self):
        y = np.array([g % 2 for g in range(50) for _ in range(2)])
        groups = np.array([f'p{g}' for g in range(50) for _ in range(2)])
        p = np.where(y, .9, .1)
        report = choose(y, groups, {'a': p, 'b': p}, {'single': ['a'], 'double': ['a', 'b']})
        self.assertEqual(report['selected']['candidate'], 'single')
        self.assertEqual(report['selected']['correct'], len(y))
        self.assertEqual(pd.DataFrame(dict(group=groups, fold=report['fold_ids'])).groupby('group').fold.nunique().max(), 1)

    def test_unexpected_predictions_rejected(self):
        y = np.tile([0, 1], 10)
        with self.assertRaises(ValueError):
            choose(y, np.arange(20), {'a': y.astype(float), 'test': y.astype(float)}, {'only': ['a']})


if __name__ == '__main__': unittest.main()
