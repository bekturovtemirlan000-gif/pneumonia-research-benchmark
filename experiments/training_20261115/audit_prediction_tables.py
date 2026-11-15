"""Independently recompute published metrics from the complete public score CSVs.

No images, weights, new fitting, threshold search or model selection are used.
Run after capture_verification.py has recovered the hash-checked exports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, confusion_matrix, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE / name).read_text(encoding='utf-8'))


def equal(actual, expected, label, tolerance=1e-10):
    if not np.isfinite([actual, expected]).all() or abs(actual - expected) > tolerance:
        raise ValueError('Prediction-table mismatch: ' + label)


def metrics(frame, fresh=False):
    score = 'fresh_cpu_probability' if fresh else 'probability'
    prediction = 'fresh_cpu_prediction' if fresh else 'prediction'
    y, p, pred = [frame[k].to_numpy() for k in ['y_true', score, prediction]]
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
                accuracy=float(np.mean(y == pred)), sensitivity=float(tp/(tp+fn)),
                specificity=float(tn/(tn+fp)), brier_score=float(np.mean((y-p)**2)),
                log_loss=float(log_loss(y, p, labels=[0, 1])),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def compare_metrics(frame, expected, label, fresh=False):
    actual = metrics(frame, fresh=fresh)
    for key, value in expected.items():
        equal(actual[key], value, label + ' ' + key)
    return actual


def table(meta, images, groups):
    path = HERE / 'predictions' / meta['filename']
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Prediction CSV bytes changed: ' + meta['filename'])
    frame = pd.read_csv(path, dtype={'sha256': str, 'patient': str}, float_precision='round_trip')
    if list(frame.columns) != meta['columns'] or len(frame) != images or images != meta['rows']:
        raise ValueError('Wrong columns or incomplete table')
    if frame.isna().any().any() or frame.sha256.duplicated().any() or frame.patient.nunique() != groups:
        raise ValueError('Invalid identity or group coverage')
    if not frame.sha256.str.fullmatch('[0-9a-f]{64}').all() or not frame.sha256.is_monotonic_increasing:
        raise ValueError('Invalid sorted image hashes')
    if set(frame.y_true) != {0, 1} or not set(frame.prediction).issubset({0, 1}):
        raise ValueError('Invalid class labels')
    if frame.groupby('patient').y_true.nunique().max() != 1:
        raise ValueError('Inconsistent proxy-group labels')
    for prefix in ['', 'fresh_cpu_'] if 'fresh_cpu_probability' in frame else ['']:
        p, pred = frame[prefix+'probability'], frame[prefix+'prediction']
        if not np.isfinite(p).all() or not p.between(0, 1).all():
            raise ValueError('Invalid model probabilities')
        if not np.array_equal(pred, p >= frame.threshold):
            raise ValueError('Stored decisions differ from frozen cutoffs')
    if 'fold' in frame and (set(frame.fold) != {0, 1, 2} or frame.groupby('patient').fold.nunique().max() != 1):
        raise ValueError('Invalid outer-fold membership')
    return frame


def check_cpu(frame, expected, label):
    error = np.abs(frame.probability - frame.fresh_cpu_probability)
    changed = int(np.sum(frame.prediction != frame.fresh_cpu_prediction))
    equal(float(error.max()), expected['max_absolute_score_difference'], label + ' maximum CPU error')
    equal(changed, expected['changed_predictions'], label + ' changed CPU decisions', 0)
    passed = bool(error.max() <= 1e-4 and changed == 0)
    if passed != expected['passed']:
        raise ValueError('Strict CPU pass/fail differs')
    return dict(max_absolute_score_difference=float(error.max()), changed_predictions=changed, passed=passed)


def check_diagnostics(frame, expected):
    bins = np.minimum((frame.probability.to_numpy()*10).astype(int), 9)
    total = 0.0
    for index, record in enumerate(expected['fixed_equal_width_bins']):
        part = frame.loc[bins == index]
        equal(len(part), record['count'], 'fixed-bin count', 0)
        if len(part):
            mean, observed = float(part.probability.mean()), float(part.y_true.mean())
            equal(mean, record['mean_score'], 'bin mean')
            equal(observed, record['observed_fraction'], 'bin observed fraction')
            total += len(part)*abs(mean-observed)
        elif record['mean_score'] is not None or record['observed_fraction'] is not None:
            raise ValueError('Empty bin carries invented values')
    equal(total/len(frame), expected['expected_calibration_error'], 'fixed-bin ECE')
    for label, name in [(0, 'normal'), (1, 'pneumonia')]:
        p = frame.loc[frame.y_true == label, 'probability']
        record = expected['class_scores'][name]
        equal(len(p), record['images'], 'class count', 0)
        equal(float(p.mean()), record['mean_score'], 'class mean')
        for actual, reference in zip(np.quantile(p, [.05, .25, .5, .75, .95]), record['quantiles_5_25_50_75_95']):
            equal(actual, reference, 'class quantile')


def primary_audit():
    report, summary = read('cpu_verification.json'), read('summary.json')
    frames, results = {}, {}
    for split, n, groups in [('calibration', 482, 268), ('test', 618, 426)]:
        frame = table(report['public_prediction_exports'][split], n, groups)
        if not np.all(frame.threshold == summary['threshold']):
            raise ValueError('Primary cutoff changed')
        fresh = compare_metrics(frame, report['partitions'][split]['fresh_metrics'], split + ' fresh', fresh=True)
        check_diagnostics(frame, report['descriptive_score_diagnostics'][split])
        results[split] = dict(images=n, groups=groups, stored_metrics=metrics(frame), fresh_metrics=fresh,
                             cpu_reproduction=check_cpu(frame, report['partitions'][split], split))
        frames[split] = frame
    if set(frames['calibration'].sha256) & set(frames['test'].sha256) or set(frames['calibration'].patient) & set(frames['test'].patient):
        raise ValueError('Calibration/test overlap')
    compare_metrics(frames['test'], summary['test'], 'primary test')
    equal(int(np.sum(frames['calibration'].prediction == frames['calibration'].y_true)),
          summary['calibration_in_sample_correct'], 'calibration correct count', 0)
    descriptive = frames['test'].copy()
    descriptive['prediction'] = (descriptive.probability >= .5).astype(int)
    compare_metrics(descriptive, summary['test_at_0_5'], 'prespecified descriptive 0.5')
    # Reproduce the already-prespecified intervals; never fit a new rule.
    test = frames['test']
    original = None
    if (HERE / 'comparator_export.json').exists():
        export_record = read('comparator_export.json')
        original = table(export_record['public_prediction_exports']['original_320'], 618, 426)
        if not original[['sha256', 'patient', 'y_true']].equals(test[['sha256', 'patient', 'y_true']]):
            raise ValueError('Original/new comparator pairing differs')
        if not np.all(original.threshold == export_record['frozen_configuration']['threshold']):
            raise ValueError('Original cutoff changed')
        compare_metrics(original, summary['uncertainty']['paired_comparison']['baseline'], 'original comparator')
        paired_keys = summary['uncertainty']['paired_comparison']['difference']
        original_metrics, current_metrics = metrics(original), metrics(test)
        for key, value in paired_keys.items():
            equal(current_metrics[key]-original_metrics[key], value, 'paired point difference ' + key)
    groups = [np.flatnonzero(test.patient.to_numpy() == group) for group in test.patient.unique()]
    rng, samples, differences = np.random.default_rng(42), [], []
    for _ in range(1000):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = test.iloc[indices]
        if sample.y_true.nunique() == 2:
            current = metrics(sample)
            samples.append(current)
            if original is not None:
                baseline = metrics(original.iloc[indices])
                differences.append({key: current[key]-baseline[key] for key in paired_keys})
    uncertainty = summary['uncertainty']
    equal(len(samples), uncertainty['bootstrap_replicates'], 'bootstrap replicate count', 0)
    intervals = {}
    for key, expected in uncertainty['benchmark_cluster_bootstrap_95ci'].items():
        intervals[key] = np.percentile([sample[key] for sample in samples], [2.5, 97.5]).tolist()
        for value, reference in zip(intervals[key], expected):
            equal(value, reference, 'single-model bootstrap ' + key)
    results['single_model_group_bootstrap_reproduced'] = dict(replicates=len(samples), seed=42, intervals=intervals)
    if original is not None:
        comparison = uncertainty['paired_comparison']
        equal(len(differences), comparison['bootstrap_replicates'], 'paired replicate count', 0)
        paired_intervals = {}
        for key, expected in comparison['paired_cluster_bootstrap_95ci'].items():
            paired_intervals[key] = np.percentile([row[key] for row in differences], [2.5, 97.5]).tolist()
            for value, reference in zip(paired_intervals[key], expected):
                equal(value, reference, 'paired bootstrap ' + key)
        results['paired_group_bootstrap_reproduced'] = dict(replicates=len(differences), seed=42, intervals=paired_intervals)
    results['scope'] = ('Complete calibration and test scores; all reported test and fresh CPU metrics, decisions, score summaries and single-model group-bootstrap intervals independently recomputed. '
                        + ('Original comparator scores also verify every paired interval. ' if original is not None else 'Original comparator scores not yet available locally; paired intervals remain Kaggle-derived. ')
                        + 'No cutoff or model changed.')
    return results


def cv_audit():
    report, summary = read('cv_cpu_verification.json'), read('cv_summary.json')
    geometry = table(report['public_prediction_exports']['geometry'], 3815, 2153)
    geometry_result = compare_metrics(geometry, report['geometry_control']['pooled_metrics'], 'geometry pooled')
    for fold in report['geometry_control']['folds']:
        part = geometry.loc[geometry.fold == fold['fold']]
        if len(part) != fold['images']: raise ValueError('Geometry fold count differs')
        if not np.all(part.threshold == fold['config']['threshold']): raise ValueError('Geometry cutoff changed')
        compare_metrics(part, fold['outer_metrics'], 'geometry fold')
    results = {'geometry': geometry_result}
    for recipe, expected in summary['recipes'].items():
        frame = table(report['public_prediction_exports'][recipe], 3815, 2153)
        if not frame[['sha256', 'patient', 'y_true', 'fold']].equals(geometry[['sha256', 'patient', 'y_true', 'fold']]):
            raise ValueError('Recipe/control cohort mismatch')
        digest = hashlib.sha256('\n'.join(frame.sha256).encode()).hexdigest()
        if digest != '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d':
            raise ValueError('Original training cohort changed')
        stored = compare_metrics(frame, expected['pooled_metrics'], recipe + ' pooled')
        fresh = compare_metrics(frame, report['pooled'][recipe]['fresh_metrics'], recipe + ' fresh pooled', fresh=True)
        equal(stored['tn']+stored['tp'], expected['correct'], recipe+' correct count', 0)
        folds = []
        for fold in [f for f in summary['folds'] if f['recipe'] == recipe]:
            part = frame.loc[frame.fold == fold['fold']]
            if len(part) != fold['outer_images']: raise ValueError('Neural fold count differs')
            cutoff_check = check_neural_cutoff_serialization(part, fold['threshold'])
            compare_metrics(part, fold['outer_metrics'], recipe + ' fold')
            cpu = next(r for r in report['fold_reports'] if r['recipe'] == recipe and r['fold'] == fold['fold'])
            fold_check = check_cpu(part, cpu, recipe + ' fold')
            fold_check['frozen_cutoff_check'] = cutoff_check
            fold_check['fold'] = fold['fold']
            folds.append(fold_check)
        n_correct, g_correct = frame.prediction == frame.y_true, geometry.prediction == frame.y_true
        paired = dict(neural_only_correct=int(np.sum(n_correct & ~g_correct)),
                      geometry_only_correct=int(np.sum(~n_correct & g_correct)),
                      both_correct=int(np.sum(n_correct & g_correct)), neither_correct=int(np.sum(~n_correct & ~g_correct)),
                      accuracy_difference=float(n_correct.mean()-g_correct.mean()))
        for key, value in report['pooled'][recipe]['geometry_paired_correctness'].items():
            equal(paired[key], value, recipe + ' paired ' + key)
        results[recipe] = dict(images=len(frame), stored_metrics=stored, fresh_metrics=fresh, cpu_folds=folds,
                               geometry_paired_correctness=paired)
    results['scope'] = 'All four complete OOF tables, pooled and fold metrics, paired correctness and strict CPU comparisons independently recomputed. No model promotion.'
    return results


def check_neural_cutoff_serialization(frame, frozen):
    """Trace the existing GPU-CSV -> pandas high parser -> public-CSV path.

    The frozen JSON cutoff remains authoritative. No numerical tolerance is
    applied to the metadata check, and both saved and fresh decisions must also
    agree with the original JSON value, including boundary cases.
    """
    parsed = float(pd.read_csv(io.StringIO('threshold\n' + repr(frozen) + '\n'),
                               float_precision='high').iloc[0, 0])
    if not np.isfinite(frozen) or not np.all(frame.threshold == parsed):
        raise ValueError('Neural fold cutoff does not reproduce its CSV parser path')
    for prefix in ['', 'fresh_cpu_']:
        if not np.array_equal(frame[prefix+'prediction'], frame[prefix+'probability'] >= frozen):
            raise ValueError('Decision differs at the authoritative frozen JSON cutoff')
    return dict(frozen_json_cutoff=frozen, exported_csv_cutoff=parsed,
                difference=parsed-frozen, parser_path_reproduced_exactly=True,
                stored_and_fresh_decisions_match_frozen_json=True,
                explanation='GPU scores were written as CSV and reread with pandas default high parser before public export; original JSON rule and exact public CSV bytes are unchanged')


def audit(kind):
    result = dict(status='passed', kind=kind, checked_utc=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
                  results=primary_audit() if kind == 'primary' else cv_audit())
    (HERE / (kind + '_prediction_audit.json')).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], kind=kind, csv_metrics_independently_recomputed=True)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['primary', 'cv'], required=True)
    audit(parser.parse_args().kind)
