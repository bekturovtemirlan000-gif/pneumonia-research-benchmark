"""Reconstruct a completed benchmark's selection and frozen test predictions."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from validate_artifacts import validate


def close(a, b, description, tolerance=1e-8):
    if not math.isfinite(float(a)) or not math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=tolerance):
        raise ValueError('Inconsistent ' + description)


def scores(path, expected, threshold):
    frame = pd.read_csv(path).sort_values('sha256').reset_index(drop=True)
    if len(frame) != len(expected) or frame.sha256.duplicated().any():
        raise ValueError('Duplicate or missing predictions: ' + str(path))
    for field in ['sha256', 'patient', 'path']:
        if not frame[field].equals(expected[field]):
            raise ValueError('Prediction identity mismatch: ' + field)
    if not np.array_equal(frame.y_true, expected.label):
        raise ValueError('Prediction labels mismatch')
    p = frame.probability.to_numpy(float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probability')
    if not np.array_equal(frame.prediction, (p >= threshold).astype(int)):
        raise ValueError('Prediction disagrees with threshold')
    return p


def check_metrics(y, p, cutoff, reported):
    predicted = p >= cutoff
    tn = int(((y == 0) & ~predicted).sum()); fp = int(((y == 0) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum()); tp = int(((y == 1) & predicted).sum())
    actual = dict(auroc=roc_auc_score(y,p), average_precision=average_precision_score(y,p),
                  accuracy=np.mean(y==predicted), sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                  brier_score=np.mean((y-p)**2), log_loss=log_loss(y,p,labels=[0,1]),
                  tn=tn, fp=fp, fn=fn, tp=tp)
    for name,value in actual.items():
        close(reported[name],value,name)


def temperature_scale(p, temperature):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Invalid temperature')
    return expit(logit(np.clip(p,1e-7,1-1e-7))/temperature)


def validate_benchmark(directory):
    root=Path(directory)
    basic=validate(root)
    manifest=pd.read_csv(root/'split_manifest.csv')
    frames={k:f.sort_values('sha256').reset_index(drop=True) for k,f in manifest.groupby('split')}
    config=json.loads((root/'inference_config.json').read_text())
    results=json.loads((root/'all_trial_results.json').read_text())
    expected={(a,s) for a in ['efficientnet_b0','densenet121','resnet18'] for s in [42,123,2026]}
    if len(results)!=9 or {(r['architecture'],r['seed']) for r in results}!=expected:
        raise ValueError('Incomplete or duplicate architecture/seed trials')
    selection_by_arch={a:[] for a,_ in expected}; raw_test={}; raw_selection={}
    for record in results:
        trial=f"{record['architecture']}_seed_{record['seed']}"
        if record['trial']!=trial:
            raise ValueError('Unexpected trial identifier')
        folder=root/trial
        if not (folder/'best_weights.pt').is_file():
            raise ValueError('Missing trial weights: '+trial)
        history=pd.read_csv(folder/'history.csv')
        if len(history)!=record['epochs_completed'] or list(history.epoch)!=list(range(1,len(history)+1)):
            raise ValueError('Inconsistent epoch history')
        best=-1.; chosen=None
        for row in history.itertuples():
            if row.selection_auroc>best+1e-5:
                best=row.selection_auroc; chosen=row.epoch
        if chosen!=record['best_epoch']:
            raise ValueError('Checkpoint selection does not follow recorded rule')
        ps=scores(folder/'selection_predictions.csv',frames['selection'],.5)
        auc=float(roc_auc_score(frames['selection'].label,ps))
        close(auc,record['selection_auroc'],'selection AUROC')
        close(auc,best,'checkpoint AUROC',tolerance=1e-5)
        selection_by_arch[record['architecture']].append(auc)
        raw_selection[trial]=ps
        p=scores(folder/'test_predictions_raw.csv',frames['test'],record['raw_threshold'])
        q=scores(folder/'test_predictions.csv',frames['test'],record['threshold'])
        if not np.allclose(q,temperature_scale(p,record['temperature']),atol=1e-10,rtol=1e-8):
            raise ValueError('Trial calibration mismatch')
        y=frames['test'].label.to_numpy()
        check_metrics(y,p,record['raw_threshold'],record['raw_test'])
        check_metrics(y,q,record['threshold'],record['calibrated_test'])
        check_metrics(y,p,.5,record['raw_test_at_0_5'])
        raw_test[trial]=p
    means={a:float(np.mean(values)) for a,values in selection_by_arch.items()}
    chosen=sorted(means,key=lambda a:(-means[a],a))[0]
    if config['architecture']!=chosen:
        raise ValueError('Architecture selection mismatch')
    trials=[f'{chosen}_seed_{s}' for s in [42,123,2026]]
    if set(config['selected_checkpoints'])!={t+'/best_weights.pt' for t in trials}:
        raise ValueError('Selected ensemble membership mismatch')
    for architecture,mean in means.items():
        close(config['selection_architecture_mean_auroc'][architecture],mean,'architecture mean')
    for split,source in [('selection',raw_selection),('test',raw_test)]:
        expected_p=temperature_scale(np.mean([source[t] for t in trials],axis=0),config['temperature'])
        actual_p=scores(root/f'{split}_predictions.csv',frames[split],config['threshold'])
        if not np.allclose(actual_p,expected_p,atol=1e-10,rtol=1e-8):
            raise ValueError('Ensemble reconstruction mismatch: '+split)
    metrics=json.loads((root/'metrics.json').read_text())
    check_metrics(frames['test'].label.to_numpy(),actual_p,config['threshold'],metrics['test'])
    return dict(basic, trials_verified=9, selected_architecture=chosen,
                ensemble_reconstructed=True, all_reported_test_metrics_verified=True,
                limitation='Verifies internal artifact consistency, not independent patient identity or clinical validity')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate_benchmark(args.directory),indent=2))
