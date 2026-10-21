import tempfile
import unittest
from pathlib import Path

import pandas as pd

from inspect_frozen_errors import inspect,subgroup


class FrozenErrorTests(unittest.TestCase):
    def fixture(self,root):
        a=root/'a';b=root/'b';a.mkdir();b.mkdir()
        frame=pd.DataFrame(dict(sha256=['a','b','c','d'],patient=['n1','n2','p1','p2'],
            path=['normal1.jpeg','normal2.jpeg','person1_bacteria_1.jpeg','unknown.jpeg'],
            y_true=[0,0,1,1],probability=[.8,.2,.8,.2],prediction=[1,0,1,0]))
        frame.to_csv(a/'test_predictions.csv',index=False)
        other=frame.copy();other['prediction']=[0,1,1,0];other['probability']=[.2,.8,.8,.2]
        other.iloc[::-1].to_csv(b/'test_predictions.csv',index=False)
        return a,b

    def test_alignment_overlap_and_unspecified_subgroup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b=self.fixture(root)
            report=inspect(a,b,root/'out')
            self.assertEqual(report['overlap'],dict(both_correct=1,both_incorrect=1,
                only_224_correct=1,only_320_correct=1,different_binary_predictions=2))
            unknown=report['subgroups']['320']['pneumonia_unspecified_filename']
            self.assertEqual(unknown['fn'],1);self.assertIsNone(unknown['specificity'])
            self.assertEqual(len(pd.read_csv(root/'out/frozen_error_cases.csv')),3)
            self.assertEqual(report['overlap_by_source_label']['0']['only_224_correct'],1)
            self.assertEqual(report['overlap_by_source_label']['1']['both_incorrect'],1)
            self.assertEqual(report['proxy_groups_spanning_filename_subgroups'],0)

    def test_changed_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a,b=self.fixture(root)
            frame=pd.read_csv(b/'test_predictions.csv');frame.loc[0,'patient']='changed'
            frame.to_csv(b/'test_predictions.csv',index=False)
            with self.assertRaisesRegex(ValueError,'cohort mismatch'):
                inspect(a,b,root/'out')

    def test_filename_labels_do_not_override_binary_source_label(self):
        self.assertEqual(subgroup('person1_virus_2.jpeg',1),'pneumonia_virus_filename')
        self.assertEqual(subgroup('person1_virus_2.jpeg',0),'normal_source_label')


if __name__=='__main__':
    unittest.main()
