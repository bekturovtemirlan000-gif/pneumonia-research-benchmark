"""Paired descriptive comparison of frozen predictions; never fits thresholds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def metrics(frame):
    y = frame.y_true.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    pred = frame.prediction.to_numpy(dtype=int)
    return {
        'auroc': float(roc_auc_score(y, p)),
        'average_precision': float(average_precision_score(y, p)),
        'accuracy': float(np.mean(y == pred)),
        'sensitivity': float(np.mean(pred[y == 1] == 1)),
        'specificity': float(np.mean(pred[y == 0] == 0)),
        'brier_score': float(np.mean((y - p) ** 2)),
    }


def compare(baseline_dir, followup_dir, replicates=1000):
    if replicates < 1:
        raise ValueError('replicates must be positive')
    frames = []
    for directory in (baseline_dir, followup_dir):
        frame = pd.read_csv(Path(directory) / 'test_predictions.csv')
        required = {'sha256', 'patient', 'y_true', 'probability', 'prediction'}
        if not required.issubset(frame.columns) or frame.sha256.duplicated().any():
            raise ValueError('Missing fields or duplicate image hashes')
        if set(frame.y_true) != {0, 1} or not set(frame.prediction).issubset({0, 1}):
            raise ValueError('Expected binary labels with both classes represented')
        if not np.isfinite(frame.probability).all() or not frame.probability.between(0, 1).all():
            raise ValueError('Invalid probabilities')
        if frame.patient.isna().any() or frame.sha256.isna().any():
            raise ValueError('Missing pairing or cluster identifiers')
        frames.append(frame.sort_values('sha256').reset_index(drop=True))
    a, b = frames
    for column in ('sha256', 'patient', 'y_true'):
        if not a[column].equals(b[column]):
            raise ValueError('Test cohorts or labels differ: ' + column)
    ma, mb = metrics(a), metrics(b)
    groups = [np.flatnonzero(a.patient.to_numpy() == group) for group in a.patient.unique()]
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(replicates):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if a.iloc[idx].y_true.nunique() != 2:
            continue
        sa, sb = metrics(a.iloc[idx]), metrics(b.iloc[idx])
        differences.append({key: sb[key] - sa[key] for key in sa})
    if not differences:
        raise ValueError('No bootstrap sample contained both classes')
    return {
        'status': 'Exploratory paired comparison; original test results already inspected',
        'direction': 'followup minus baseline; lower is better for Brier score only',
        'images': len(a), 'filename_proxy_groups': len(groups),
        'baseline': ma, 'followup': mb,
        'difference': {key: mb[key] - ma[key] for key in ma},
        'paired_cluster_bootstrap_95ci': {
            key: np.percentile([d[key] for d in differences], [2.5, 97.5]).tolist()
            for key in ma},
        'bootstrap_replicates': len(differences),
        'limitations': 'Filename groups are proxies. Intervals describe test sampling, not training variation, model selection uncertainty or external generalization. No multiplicity adjustment. Thresholds remain those saved by each experiment.'
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('followup', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    report = compare(args.baseline, args.followup, args.replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
