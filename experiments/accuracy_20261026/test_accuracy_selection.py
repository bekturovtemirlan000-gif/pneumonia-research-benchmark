import unittest
import numpy as np
from accuracy_selection import accuracy_threshold, candidates, choose


class AccuracySelectionTests(unittest.TestCase):
    def test_midpoint_and_accuracy(self):
        threshold, correct = accuracy_threshold([0, 0, 1, 1], [.1, .3, .7, .9])
        self.assertEqual(threshold, .5)
        self.assertEqual(correct, 4)

    def test_tie_preserves_more_sensitivity(self):
        threshold, correct = accuracy_threshold([0, 1], [.5, .5])
        self.assertEqual(threshold, 0.)
        self.assertEqual(correct, 1)

    def test_invalid_scores_rejected(self):
        for values in [[np.nan, .3], [-.1, .3], [.1, 1.1]]:
            with self.assertRaises(ValueError): accuracy_threshold([0, 1], values)

    def test_groups_stay_together_and_original_wins_tie(self):
        y = np.repeat([0, 1] * 20, 2)
        groups = np.repeat([f'g{i}' for i in range(40)], 2)
        scores = {name: .1 + .8*y for name in candidates()['both_all_architectures']}
        report = choose(y, groups, scores)
        self.assertEqual(len(report['candidates']), 12)
        self.assertEqual(report['selected']['candidate'], '320_efficientnet_b0')
        self.assertEqual(report['selected']['oof_accuracy'], 1.)
        for group in set(groups):
            self.assertEqual(len(set(np.asarray(report['fold_ids'])[groups == group])), 1)
        self.assertFalse(report['test_predictions_accessed'])

    def test_selection_does_not_accept_test_scores(self):
        y = np.array([0, 1] * 20)
        groups = np.array([f'g{i}' for i in range(40)])
        scores = {name: .1 + .8*y for name in candidates()['both_all_architectures']}
        scores['test_scores'] = np.ones(40)
        with self.assertRaises(ValueError): choose(y, groups, scores)


if __name__ == '__main__': unittest.main()
