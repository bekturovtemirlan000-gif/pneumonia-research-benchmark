"""Development-only decisions for the November training experiment."""
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


def internal_split(frame):
    """Reserve every fifth hash-ordered group in each class for checkpoint selection."""
    if frame.groupby('patient').label.nunique().max() != 1:
        raise ValueError('Mixed-label proxy group')
    held = set()
    for label in [0, 1]:
        groups = sorted(frame.loc[frame.label == label, 'patient'].unique(),
                        key=lambda g: hashlib.sha256(('20261115:' + g).encode()).hexdigest())
        held.update(groups[::5])
    fit = frame.loc[~frame.patient.isin(held)].copy().reset_index(drop=True)
    stop = frame.loc[frame.patient.isin(held)].copy().reset_index(drop=True)
    if set(fit.patient) & set(stop.patient) or set(fit.sha256) & set(stop.sha256):
        raise ValueError('Internal split overlap')
    if set(fit.label) != {0, 1} or set(stop.label) != {0, 1}:
        raise ValueError('Both classes required')
    return fit, stop


def threshold(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(y) != {0, 1}:
        raise ValueError('Aligned binary labels required')
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probabilities')
    values = np.unique(p)
    cuts = np.r_[0., values[:-1] + np.diff(values) / 2, np.nextafter(values[-1], np.inf)]
    counts = np.array([np.sum((p >= cut) == y) for cut in cuts])
    best = int(np.argmax(counts))
    return float(cuts[best]), int(counts[best])


def choose(y, groups, scores, definitions):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.shape != groups.shape or len(set(groups)) < 5:
        raise ValueError('Aligned groups required')
    required = set().union(*map(set, definitions.values()))
    if set(scores) != required:
        raise ValueError('Unexpected or missing member scores')
    for p in scores.values():
        threshold(y, p)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261115)
                 .split(np.zeros(len(y)), y, groups))
    fold_ids = np.full(len(y), -1, dtype=int)
    records = []
    for number, (fit, valid) in enumerate(folds):
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError('Cross-fold group overlap')
        if set(y[fit]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError('Each fold needs both classes')
        fold_ids[valid] = number
    for name, members in definitions.items():
        p = np.mean([scores[m] for m in members], axis=0)
        pred = np.empty(len(y), dtype=int)
        cuts = []
        for fit, valid in folds:
            cut, _ = threshold(y[fit], p[fit])
            pred[valid] = p[valid] >= cut
            cuts.append(cut)
        records.append(dict(candidate=name, members=members, correct=int(np.sum(pred == y)),
                            images=len(y), cross_fitted_accuracy=float(np.mean(pred == y)),
                            auroc=float(roc_auc_score(y, p)), fold_thresholds=cuts))
    records.sort(key=lambda r: (-r['correct'], len(r['members']), r['candidate']))
    return dict(selected=records[0], candidates=records, fold_ids=fold_ids.tolist(),
                limitation='Development estimate after candidate selection; not nested CV. Prior experiments inspected these cohorts.')
