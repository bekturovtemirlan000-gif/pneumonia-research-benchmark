import tempfile
import unittest
import json
import hashlib
import contextlib
import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from verify_calibration_fit import calibration_frame, verify


class CalibrationIdentityTests(unittest.TestCase):
    def test_verification_io_supports_legacy_and_explicit_resolution(self):
        class MeanModel(torch.nn.Module):
            def forward(self,x): return x.mean(dim=(1,2,3)).reshape(-1,1)
        for size in (224,320):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);model_dir=root/'model';model_dir.mkdir()
                data=root/'data';rows=[]
                for label,value in ((0,20),(1,230)):
                    folder=data/'train'/('NORMAL' if label==0 else 'PNEUMONIA');folder.mkdir(parents=True)
                    image=Image.new('L',(8,8),value);name=f'image{label}.png';image.save(folder/name)
                    rows.append(dict(original_split='train',filename=name,y_true=label,
                        sha256=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
                        probability=.1 if label==0 else .9,prediction=label))
                torch.save({},model_dir/'weights.pt')
                config=dict(architecture='resnet18',classes=['NORMAL','PNEUMONIA'],
                    selected_checkpoints=['weights.pt'],temperature=1.,threshold=.5)
                if size==320: config['image_size']=size
                (model_dir/'inference_config.json').write_text(json.dumps(config))
                with patch('verify_calibration_fit.calibration_frame',return_value=pd.DataFrame(rows)), \
                     patch('verify_calibration_fit.model_for',return_value=MeanModel()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    report=verify(model_dir,data,root/'output')
                self.assertEqual(report['image_size'],size)
                self.assertEqual(report['images'],2)
                self.assertFalse(report['test_predictions_accessed'])
                self.assertEqual(json.loads((model_dir/'inference_config.json').read_text()),config)
                self.assertTrue((root/'output/calibration_cpu_scores.csv').is_file())

    def test_predictions_without_split_column_match_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=pd.DataFrame([dict(path=f'image/{i}',y_true=i%2,patient=f'person{i}',sha256=f'{i:064x}') for i in range(482)])
            frame.to_csv(root/'calibration_predictions.csv',index=False)
            manifest=frame.rename(columns={'y_true':'label'}).assign(split='calibration')
            manifest.to_csv(root/'split_manifest.csv',index=False)
            self.assertEqual(len(calibration_frame(root)),482)
            # Same row count with a changed identity must still be rejected.
            frame.loc[0,'patient']='different-person'
            frame.to_csv(root/'calibration_predictions.csv',index=False)
            with self.assertRaises(ValueError): calibration_frame(root)


if __name__=='__main__': unittest.main()
