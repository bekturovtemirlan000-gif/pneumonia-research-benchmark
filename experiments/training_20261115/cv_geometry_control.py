"""Prespecified dimension-only comparator on the same three outer folds."""
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from selection import internal_split, threshold
from research_benchmark import metrics
from compare_experiments import metrics as pooled_metrics


def run(frame, folds, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    def x(values):
        dimensions = values[['width', 'height']].to_numpy(dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('Invalid image dimensions')
        return np.log(dimensions)
    frozen = []
    for fold, (outer_train, outer_test) in enumerate(folds):
        fit, stop = internal_split(frame.iloc[outer_train].reset_index(drop=True))
        held = frame.iloc[outer_test].reset_index(drop=True)
        if set(held.patient) & (set(fit.patient) | set(stop.patient)):
            raise ValueError('Geometry control group overlap')
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000,
                                                                   class_weight=None, random_state=42))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x(fit), fit.label)
        inner = model.predict_proba(x(stop))[:, 1]
        cut, count = threshold(stop.label, inner)
        scaler, classifier = model.named_steps['standardscaler'], model.named_steps['logisticregression']
        config = dict(fold=fold, threshold=cut, internal_correct=count, fit_images=len(fit), inner_images=len(stop),
                      features=['natural_log_width', 'natural_log_height'], C=1.0, solver='lbfgs', max_iter=1000,
                      class_weight=None, standardization_mean=scaler.mean_.tolist(),
                      standardization_scale=scaler.scale_.tolist(), coefficient=classifier.coef_[0].tolist(),
                      intercept=float(classifier.intercept_[0]), classes=classifier.classes_.tolist(),
                      iterations=int(classifier.n_iter_[0]))
        path = output / f'geometry_fold_{fold}_frozen.json'
        path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        frozen.append((model, held, config, hashlib.sha256(path.read_bytes()).hexdigest()))
    records, pieces = [], []
    for model, held, config, digest in frozen:
        probability = model.predict_proba(x(held))[:, 1]
        standardized = (x(held) - np.asarray(config['standardization_mean'])) / np.asarray(config['standardization_scale'])
        logit = standardized @ np.asarray(config['coefficient']) + config['intercept']
        exported = 1 / (1 + np.exp(-logit))
        if np.max(np.abs(exported - probability)) > 1e-12:
            raise ValueError('Exported geometry coefficients fail reproduction')
        fold = config['fold']; cut = config['threshold']
        piece = held[['sha256', 'patient', 'label']].assign(y_true=held.label, probability=probability,
                 prediction=(probability >= cut).astype(int), fold=fold, threshold=cut)
        pieces.append(piece)
        records.append(dict(fold=fold, images=len(held), groups=int(held.patient.nunique()),
                            config=config, config_sha256=digest, outer_metrics=metrics(held.label, probability, cut)))
    values = pd.concat(pieces).sort_values('sha256').reset_index(drop=True)
    if not values.sha256.equals(frame.sha256): raise ValueError('Incomplete geometry OOF coverage')
    values.to_csv(output / 'geometry_oof_predictions.csv', index=False)
    report = dict(status='completed', images=len(values), groups=int(values.patient.nunique()),
                  folds=records, pooled_metrics=pooled_metrics(values), correct=int(np.sum(values.y_true == values.prediction)),
                  fold_accuracies=[r['outer_metrics']['accuracy'] for r in records],
                  configurations_frozen_before_all_outer_scoring=True, coefficient_export_reproduced=True,
                  primary_model_changed=False,
                  interpretation='Image dimensions are predictors; decoded pixels are used only for identity checking. This tests a dataset association, not whether neural models use dimensions. This is an exploratory internal comparator, not external testing.')
    (output / 'geometry_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('CV_GEOMETRY_CONTROL', json.dumps(report), flush=True)
    return report, values
