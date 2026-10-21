"""Choose the prespecified resolution/architecture using development predictions only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def select(low, high):
    candidates=[]; reference=None
    for size,root in [(224,Path(low)),(320,Path(high))]:
        config=json.loads((root/'inference_config.json').read_text())
        if config.get('image_size',224)!=size:
            raise ValueError('Resolution does not match supplied arm')
        manifest=pd.read_csv(root/'split_manifest.csv')
        development=manifest[manifest.split!='test'][['sha256','patient','label','split']].sort_values('sha256').reset_index(drop=True)
        if reference is None: reference=development
        elif not development.equals(reference):
            raise ValueError('Development cohorts differ between resolution arms')
        expected=development[development.split=='selection'].sort_values('sha256').reset_index(drop=True)
        if expected.sha256.duplicated().any() or set(expected.label)!={0,1}:
            raise ValueError('Invalid selection cohort')
        means={}
        for architecture in ['efficientnet_b0','densenet121','resnet18']:
            values=[]
            for seed in [42,123,2026]:
                trial=f'{architecture}_seed_{seed}'
                frame=pd.read_csv(root/trial/'selection_predictions.csv').sort_values('sha256').reset_index(drop=True)
                for field in ['sha256','patient']:
                    if not frame[field].equals(expected[field]):
                        raise ValueError('Selection identity mismatch: '+trial)
                if not np.array_equal(frame.y_true,expected.label):
                    raise ValueError('Selection label mismatch: '+trial)
                p=frame.probability.to_numpy(float)
                if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
                    raise ValueError('Invalid selection scores')
                values.append(float(roc_auc_score(frame.y_true,p)))
            mean=float(np.mean(values)); means[architecture]=mean
            candidates.append(dict(image_size=size,architecture=architecture,
                                   selection_auroc_by_seed=values,mean_selection_auroc=mean))
        chosen=sorted(means,key=lambda name:(-means[name],name))[0]
        if config['architecture']!=chosen:
            raise ValueError('Arm export does not follow architecture selection rule')
    ordered=sorted(candidates,key=lambda c:(-c['mean_selection_auroc'],c['image_size'],c['architecture']))
    return dict(selected=ordered[0],candidates=ordered,
                criterion='Highest mean model-selection AUROC across seeds; ties prefer 224 then alphabetical architecture',
                test_predictions_accessed=False,
                interpretation='Exploratory protocol extension; expanded selection search can increase optimism. Both arms must still be reported.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('low',type=Path); parser.add_argument('high',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=select(args.low,args.high)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
