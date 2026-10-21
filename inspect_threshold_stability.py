"""Describe development-only threshold variability without changing deployment."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def inspect(directory, replicates=1000):
    directory = Path(directory)
    split = 'calibration' if (directory/'calibration_predictions.csv').is_file() else 'validation'
    frame = pd.read_csv(directory / f'{split}_predictions.csv')
    original = json.loads((directory / 'metrics.json').read_text())['threshold']
    if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
        raise ValueError('Invalid validation scores')
    if set(frame.y_true) != {0, 1} or frame.patient.isna().any():
        raise ValueError('Both classes and patient proxies required')
    groups = [np.flatnonzero(frame.patient.to_numpy() == g) for g in frame.patient.unique()]
    rng = np.random.default_rng(12345)
    thresholds = []
    for _ in range(replicates):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = frame.iloc[idx]
        if sample.y_true.nunique() != 2:
            continue
        fpr, tpr, cutoffs = roc_curve(sample.y_true, sample.probability)
        valid = np.isfinite(cutoffs)
        thresholds.append(float(cutoffs[valid][np.argmax((tpr-fpr)[valid])]))
    if not thresholds:
        raise ValueError('No usable bootstrap samples')
    return {
        'data_used': f'{split}_predictions.csv only; no test predictions accessed',
        'frozen_deployment_threshold': original,
        'rule': 'Maximum Youden J; first maximum in sklearn roc_curve output',
        'bootstrap_replicates': len(thresholds),
        'filename_proxy_groups': len(groups),
        'threshold_percentiles_2_5_25_50_75_97_5': np.percentile(thresholds, [2.5, 25, 50, 75, 97.5]).tolist(),
        'distinct_selected_thresholds': len(set(thresholds)),
        'limitations': ('Descriptive resampling stability, not a guaranteed confidence interval for an optimal clinical threshold. '
            + ('Calibration temperature is held fixed; this omits uncertainty from fitting temperature. '
               if split == 'calibration' else 'Checkpoint selection used the same validation data. ')
            + 'Filename groups are proxies. Deployment threshold was not changed.')
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
