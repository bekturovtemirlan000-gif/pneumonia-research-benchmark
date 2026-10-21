import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from image_geometry_control import fit_control,features
from analyze_geometry_control import reconstruct


class GeometryExportTests(unittest.TestCase):
    def create(self,root):
        frame=pd.DataFrame(dict(sha256=['a','b','c','d'],patient=['a','b','c','d'],
            split=['train']*4,label=[0,0,1,1],width=[100,120,200,220],height=[100,120,200,220]))
        model,config=fit_control(frame)
        frame['geometry_score']=model.predict_proba(features(frame))[:,1]
        frame['geometry_prediction']=(frame.geometry_score>=.5).astype(int)
        (root/'geometry_model.json').write_text(json.dumps(config),encoding='utf-8')
        frame.to_csv(root/'geometry_predictions.csv',index=False)

    def test_exported_coefficients_reconstruct_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.create(root)
            frame,report=reconstruct(root)
            self.assertEqual(len(frame),4)
            self.assertEqual(report['changed_classifications'],0)
            self.assertLessEqual(report['max_absolute_score_difference'],1e-12)

    def test_changed_score_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.create(root)
            path=root/'geometry_predictions.csv';frame=pd.read_csv(path)
            frame.loc[0,'geometry_score']+=.01;frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'reconstruction failed'):
                reconstruct(root)


if __name__=='__main__':
    unittest.main()
