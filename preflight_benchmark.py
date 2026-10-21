import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from research_benchmark import make_model,split_cohorts,temperature_for,calibrated,threshold_for,metrics
old=pd.read_csv('artifacts/baseline_v1/pneumonia_outputs/split_manifest.csv')
frames=split_cohorts(old.drop(columns='split'))
assert set(frames['train'].sha256)==set(old[old.split=='train'].sha256)
assert (set(frames['selection'].sha256)|set(frames['calibration'].sha256))==set(old[old.split=='validation'].sha256)
assert set(frames['test'].sha256)==set(old[old.split=='test'].sha256)
for architecture in ['efficientnet_b0','densenet121','resnet18']:
    model,head=make_model(architecture); model.eval()
    with torch.inference_mode(): y=model(torch.zeros(2,3,224,224))
    assert y.shape==(2,1) and torch.isfinite(y).all()
    assert set(id(p) for p in head.parameters()).issubset({id(p) for p in model.parameters()})
    print('Model forward check passed:',architecture,flush=True)
    del model
labels=np.array([0,0,0,1,1,1]); probabilities=np.array([.01,.1,.8,.2,.9,.99])
t=temperature_for(labels,probabilities); scores=calibrated(probabilities,t)
assert t>0 and np.isfinite(scores).all()
assert np.array_equal(np.argsort(probabilities),np.argsort(scores))
assert metrics(labels,scores,threshold_for(labels,scores))['auroc']>0
result={'model_forward_checks':'all three passed','cohorts_match_existing_training_and_test':True,'selection_images':len(frames['selection']),'calibration_images':len(frames['calibration']),'calibration_finite_and_order_preserving':True}
Path('artifacts/research_benchmark_preflight.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
