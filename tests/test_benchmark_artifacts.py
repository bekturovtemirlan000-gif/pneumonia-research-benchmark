import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from validate_benchmark import validate_benchmark, temperature_scale


class BenchmarkArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        frames={split:pd.DataFrame([dict(path=f'{split}/{y}',label=y,patient=f'{split}-{y}',
                    sha256=f'{split}-{y}',split=split) for y in [0,1]])
                for split in ['train','selection','calibration','test']}
        pd.concat(frames.values()).to_csv(self.root/'split_manifest.csv',index=False)
        p=np.array([.2,.8]); q=temperature_scale(p,1.5)
        def reported(scores):
            return dict(auroc=1.,average_precision=1.,accuracy=1.,sensitivity=1.,specificity=1.,
                        brier_score=float(np.mean((np.array([0,1])-scores)**2)),
                        log_loss=log_loss([0,1],scores,labels=[0,1]),tn=1,fp=0,fn=0,tp=1)
        def predictions(path,split,scores):
            frames[split].assign(y_true=[0,1],probability=scores,prediction=[0,1]).to_csv(path,index=False)
        results=[]
        for arch in ['efficientnet_b0','densenet121','resnet18']:
            for seed in [42,123,2026]:
                trial=f'{arch}_seed_{seed}'; folder=self.root/trial; folder.mkdir()
                (folder/'best_weights.pt').write_bytes(b'placeholder')
                pd.DataFrame([dict(epoch=1,selection_auroc=1.)]).to_csv(folder/'history.csv',index=False)
                predictions(folder/'selection_predictions.csv','selection',p)
                predictions(folder/'test_predictions_raw.csv','test',p)
                predictions(folder/'test_predictions.csv','test',q)
                results.append(dict(trial=trial,architecture=arch,seed=seed,best_epoch=1,
                     epochs_completed=1,selection_auroc=1.,temperature=1.5,threshold=.5,raw_threshold=.5,
                     raw_test=reported(p),calibrated_test=reported(q),raw_test_at_0_5=reported(p)))
        self.write('all_trial_results.json',results)
        for split in ['selection','calibration','test']:
            predictions(self.root/f'{split}_predictions.csv',split,q)
        self.write('inference_config.json',dict(architecture='densenet121',temperature=1.5,threshold=.5,
                   classes=['NORMAL','PNEUMONIA'],
                   selected_checkpoints=[f'densenet121_seed_{s}/best_weights.pt' for s in [42,123,2026]],
                   selection_architecture_mean_auroc={a:1. for a in ['efficientnet_b0','densenet121','resnet18']}))
        self.write('metrics.json',dict(threshold=.5,temperature=1.5,test=reported(q)))

    def write(self,name,data):
        (self.root/name).write_text(json.dumps(data))

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_reconstruction(self):
        self.assertEqual(validate_benchmark(self.root)['trials_verified'],9)

    def test_incomplete_comparison_rejected(self):
        results=json.loads((self.root/'all_trial_results.json').read_text())
        self.write('all_trial_results.json',results[:-1])
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            validate_benchmark(self.root)

    def test_corrupt_ensemble_scores_rejected(self):
        path=self.root/'test_predictions.csv'; frame=pd.read_csv(path)
        frame.loc[0,'probability']=.3; frame.to_csv(path,index=False)
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            validate_benchmark(self.root)


if __name__=='__main__':
    unittest.main()
