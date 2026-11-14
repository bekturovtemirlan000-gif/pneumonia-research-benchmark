"""Check observed records and independently recompute public per-image metrics."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_prediction_tables import audit as audit_predictions

HERE = Path(__file__).resolve().parent


def read(name): return json.loads((HERE / name).read_text(encoding='utf-8'))


def close(a, b, label, tolerance=1e-10):
    if not np.isfinite([a, b]).all() or abs(a-b) > tolerance:
        raise ValueError('Inconsistent ' + label)


def check_confusion(metrics, count):
    tn, fp, fn, tp = [metrics[k] for k in ['tn', 'fp', 'fn', 'tp']]
    if any(not isinstance(v, int) or v < 0 for v in [tn, fp, fn, tp]) or tn+fp+fn+tp != count:
        raise ValueError('Invalid confusion counts')
    for name, value in [('accuracy', (tn+tp)/count), ('sensitivity', tp/(tp+fn)), ('specificity', tn/(tn+fp))]:
        close(metrics[name], value, name)
    for key in ['auroc', 'average_precision', 'brier_score']:
        if not np.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1: raise ValueError('Invalid ' + key)


def check_histories(records, histories, count_for_record):
    if len({(h['trial'], h['epoch']) for h in histories}) != len(histories): raise ValueError('Duplicate epochs')
    for record in records:
        complete = [h for h in histories if h['trial'] == record['trial'] and h['complete_epoch']]
        if len(complete) != record['epochs_completed']: raise ValueError('Incomplete captured history')
        best = min(complete, key=lambda h: (-round(h['early_stop_accuracy'] * count_for_record(record)),
                                            h['early_stop_log_loss'], h['epoch']))
        if best['epoch'] != record['best_epoch']: raise ValueError('Reported best epoch violates rule')
        if len(record['checkpoint_sha256']) != 64: raise ValueError('Missing checkpoint digest')


def audit():
    primary, record = read('summary.json'), read('run_record.json')
    if primary['status'] != 'completed' or record['status'] != 'completed': raise ValueError('Primary run not complete')
    if len(primary['trials']) != 9 or primary['trials'] != read('development_trials.json'):
        raise ValueError('Nine matching primary trial records required')
    if hashlib.sha256((HERE / 'submitted_source.py').read_text(encoding='utf-8').encode()).hexdigest() != record['submitted_source_sha256_lf']:
        raise ValueError('Primary source changed')
    for rec in primary['trials']: check_confusion(rec['selection_at_0_5'], 456)
    check_histories(primary['trials'], read('epoch_history.json'), lambda r: 781)
    check_confusion(primary['test'], 618); check_confusion(primary['test_at_0_5'], 618)
    choice = read('development_choice.json')
    candidates = choice['candidates']
    if len(candidates) != 4 or candidates != sorted(candidates, key=lambda c: (-c['correct'], len(c['members']), c['candidate'])):
        raise ValueError('Candidate ranking differs')
    for candidate in candidates: close(candidate['correct']/456, candidate['cross_fitted_accuracy'], 'cross-fitted accuracy')
    if choice['selected'] != candidates[0] or primary['selected_candidate'] != candidates[0]['candidate']:
        raise ValueError('Wrong primary candidate')
    frozen = read('frozen_model.json')
    if primary['config_sha256'] != frozen['config_sha256']: raise ValueError('Frozen config hash mismatch')
    if hashlib.sha256(json.dumps(frozen['config'], indent=2).encode()).hexdigest() != primary['config_sha256']:
        raise ValueError('Logged config cannot reconstruct saved hash')
    close(primary['threshold'], frozen['config']['threshold'], 'frozen cutoff', 0)
    comparison = primary['uncertainty']['paired_comparison']
    for key, value in comparison['difference'].items():
        close(comparison['followup'][key], primary['test'][key], 'paired follow-up ' + key)
        close(value, comparison['followup'][key] - comparison['baseline'][key], 'paired difference ' + key)
    close(comparison['baseline']['accuracy'], 540/618, 'original comparator accuracy')
    primary_cpu = read('cpu_verification.json')
    if primary_cpu['config_sha256'] != primary['config_sha256'] or len(primary_cpu['checkpoint_checks']) != 9:
        raise ValueError('CPU verification concerns another export')
    for split, n in [('calibration', 482), ('test', 618)]:
        report = primary_cpu['partitions'][split]
        if report['images'] != n or report['passed'] != (report['changed_predictions'] == 0 and report['max_absolute_score_difference'] <= 1e-4):
            raise ValueError('Strict primary CPU result inconsistent')
        bins = primary_cpu['descriptive_score_diagnostics'][split]['fixed_equal_width_bins']
        if sum(b['count'] for b in bins) != n: raise ValueError('Reliability bins incomplete')
    cv, cv_record = read('cv_summary.json'), read('cv_run_record.json')
    if cv['status'] != 'completed' or cv_record['status'] != 'completed' or len(cv['training_records']) != 27:
        raise ValueError('Grouped CV not complete')
    if cv['training_records'] != read('cv_training_records.json') or cv['folds'] != read('cv_fold_results.json'):
        raise ValueError('CV terminal record differs from individual logs')
    if hashlib.sha256((HERE / 'submitted_grouped_cv.py').read_text(encoding='utf-8').encode()).hexdigest() != cv_record['submitted_source_sha256_lf']:
        raise ValueError('Frozen CV source changed')
    if cv['primary_model_changed'] or cv['original_selection_calibration_test_scored']:
        raise ValueError('Secondary CV exceeded its scope')
    check_histories(cv['training_records'], read('cv_epoch_history.json'), lambda r: r['internal_images'])
    for fold in cv['folds']: check_confusion(fold['outer_metrics'], fold['outer_images'])
    for recipe, report in cv['recipes'].items():
        pieces = [f for f in cv['folds'] if f['recipe'] == recipe]
        if len(pieces) != 3 or sum(f['outer_images'] for f in pieces) != 3815: raise ValueError('CV fold coverage differs')
        correct = sum(f['outer_metrics']['tn'] + f['outer_metrics']['tp'] for f in pieces)
        if correct != report['correct']: raise ValueError('Pooled CV count differs')
        close(correct/3815, report['pooled_metrics']['accuracy'], 'pooled CV accuracy')
    cv_cpu = read('cv_cpu_verification.json')
    if cv_cpu['verified_checkpoints'] != 27 or len(cv_cpu['fold_reports']) != 9:
        raise ValueError('Incomplete CV reproduction')
    for fold in cv_cpu['fold_reports']:
        if fold['passed'] != (fold['changed_predictions'] == 0 and fold['max_absolute_score_difference'] <= 1e-4):
            raise ValueError('Strict CV CPU result inconsistent')
    geometry = cv_cpu['geometry_control']
    for fold in geometry['folds']: check_confusion(fold['outer_metrics'], fold['images'])
    close(geometry['correct']/3815, geometry['pooled_metrics']['accuracy'], 'geometry accuracy')
    for recipe, report in cv_cpu['pooled'].items():
        p = report['geometry_paired_correctness']
        if sum(p[k] for k in ['neural_only_correct','geometry_only_correct','both_correct','neither_correct']) != 3815:
            raise ValueError('Incomplete paired correctness counts')
        close((p['neural_only_correct']-p['geometry_only_correct'])/3815, p['accuracy_difference'], 'paired geometry difference')
        close(cv['recipes'][recipe]['pooled_metrics']['accuracy']-geometry['pooled_metrics']['accuracy'], p['accuracy_difference'], 'geometry comparator')
    prediction_checks = {kind: audit_predictions(kind)['status'] for kind in ['primary', 'cv']}
    result = dict(status='passed', checked_utc=datetime.now(timezone.utc).isoformat(), primary_fresh_neural_models=9,
                  secondary_fresh_neural_models=27, secondary_logistic_controls=3,
                  primary_cpu_status=primary_cpu['status'], grouped_cv_cpu_status=cv_cpu['status'],
                  public_prediction_metric_audits=prediction_checks,
                  scope='Compact evidence consistency, captured epoch decisions, frozen source/configuration hashes, plus complete public CSV hashes, per-image decisions, metrics and CPU differences independently recomputed locally. Weight loading and image-level fresh inference were separately executed in Kaggle. This is same-data reproducibility, not external validation.')
    (HERE / 'compact_record_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__': audit()
