"""Accuracy-oriented exploratory selection using development predictions only."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ARCHITECTURES = ['efficientnet_b0', 'densenet121', 'resnet18']
SEEDS = [42, 123, 2026]


def member_name(size, architecture, seed):
    return f'{size}/{architecture}_seed_{seed}'


def candidates():
    result = {}
    for size in [224, 320]:
        for architecture in ARCHITECTURES:
            result[f'{size}_{architecture}'] = [member_name(size, architecture, s) for s in SEEDS]
        result[f'{size}_all_architectures'] = [member_name(size, a, s) for a in ARCHITECTURES for s in SEEDS]
    for architecture in ARCHITECTURES:
        result[f'both_{architecture}'] = [member_name(z, architecture, s) for z in [224, 320] for s in SEEDS]
    result['both_all_architectures'] = [member_name(z, a, s) for z in [224, 320] for a in ARCHITECTURES for s in SEEDS]
    return result


def validate_scores(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or set(y) != {0, 1}:
        raise ValueError('Both binary classes and aligned one-dimensional scores required')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Scores must be finite and in [0,1]')
    return y, p


def accuracy_threshold(y, p):
    y, p = validate_scores(y, p)
    values = np.unique(p)
    # Midpoints avoid using an observed score as the decision boundary.
    thresholds = np.r_[0., values[:-1] + np.diff(values) / 2, np.nextafter(values[-1], np.inf)]
    correct = np.array([np.sum((p >= t) == y) for t in thresholds])
    # Sorted thresholds: ties choose the lowest, preserving greater sensitivity.
    index = int(np.argmax(correct))
    return float(thresholds[index]), int(correct[index])


def choose(y, groups, scores):
    y, groups = np.asarray(y), np.asarray(groups)
    if groups.shape != y.shape or any(str(g) in {'', 'nan', 'None'} for g in groups):
        raise ValueError('Aligned nonmissing groups required')
    definitions = candidates()
    required = set(definitions['both_all_architectures'])
    if set(scores) != required:
        raise ValueError('Exactly 18 original member selection scores required')
    for p in scores.values():
        validate_scores(y, p)
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261026).split(np.zeros(len(y)), y, groups))
    fold_ids = np.full(len(y), -1, dtype=int)
    for fold, (train, valid) in enumerate(folds):
        if set(groups[train]) & set(groups[valid]):
            raise ValueError('Group overlap within threshold cross-validation')
        if set(y[train]) != {0, 1} or set(y[valid]) != {0, 1}:
            raise ValueError('Both classes required in each fold')
        fold_ids[valid] = fold
    records = []
    for name, members in definitions.items():
        p = np.mean([scores[m] for m in members], axis=0)
        oof = np.empty(len(y), dtype=int)
        thresholds = []
        for train, valid in folds:
            t, _ = accuracy_threshold(y[train], p[train])
            oof[valid] = p[valid] >= t
            thresholds.append(t)
        records.append(dict(candidate=name, members=members, member_count=len(members),
                            correct=int(np.sum(oof == y)), images=len(y),
                            oof_accuracy=float(np.mean(oof == y)),
                            selection_auroc=float(roc_auc_score(y, p)),
                            fold_thresholds=thresholds))
    # Preserve the original model on an accuracy tie, then prefer fewer members and fixed name order.
    records.sort(key=lambda r: (-r['correct'], r['candidate'] != '320_efficientnet_b0', r['member_count'], r['candidate']))
    return dict(selected=records[0], candidates=records, fold_ids=fold_ids.tolist(),
                criterion='Pooled five-fold grouped out-of-fold threshold accuracy on selection only',
                tie_rule='Original 320 EfficientNet-B0 candidate, then fewer members, then name',
                test_predictions_accessed=False,
                limitation='Threshold cross-fitting only: original checkpoint selection used this selection cohort. This is not unbiased nested validation of training or candidate selection.')
