"""Describe three-seed variation in completed trials without selecting or refitting models."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def summarize(root=Path('.')):
    root = Path(root)
    rows, inputs = [], {}
    architectures = {'efficientnet_b0', 'densenet121', 'resnet18'}
    for size in (224, 320):
        path = root / f'results/benchmark{size}_all_trial_results.json'
        trials = json.loads(path.read_text(encoding='utf-8'))
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = {(architecture, seed) for architecture in architectures for seed in (42, 123, 2026)}
        if len(trials) != 9 or {(r['architecture'], r['seed']) for r in trials} != expected:
            raise ValueError('Exactly nine planned architecture/seed trials required')
        for architecture in sorted(architectures):
            members = sorted((r for r in trials if r['architecture'] == architecture), key=lambda r: r['seed'])
            metrics = {}
            values = {'selection_auroc': [r['selection_auroc'] for r in members]}
            for metric in ('auroc', 'average_precision', 'brier_score', 'accuracy', 'sensitivity', 'specificity'):
                values['individual_calibrated_test_' + metric] = [r['calibrated_test'][metric] for r in members]
            for metric, observations in values.items():
                if not all(math.isfinite(v) for v in observations):
                    raise ValueError('Nonfinite trial measure')
                metrics[metric] = dict(mean=statistics.mean(observations),
                    sample_standard_deviation=statistics.stdev(observations),
                    minimum=min(observations), maximum=max(observations), values_in_seed_order=observations)
            rows.append(dict(image_size=size, architecture=architecture, seeds=[r['seed'] for r in members],
                             completed_epochs=[r['epochs_completed'] for r in members], metrics=metrics))
    report = dict(status='Post hoc descriptive summary of all completed seeds; no model choice',
                  input_sha256=inputs, groups=rows,
                  interpretation='Each row describes three individually calibrated models on the same fixed cohorts. Means are not ensemble performance; sample SD is not a confidence interval.',
                  limitations=['Three seeds provide limited information about optimization variability.',
                               'Shared data and development selection mean these are not independent population samples.',
                               'Training variation, test-sampling uncertainty and selection optimism are distinct.',
                               'No ranking-based replacement of the frozen development-selected model, no significance tests or threshold changes.'])
    (root / 'results/seed_variation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        a = row['metrics']['individual_calibrated_test_auroc']
        b = row['metrics']['individual_calibrated_test_brier_score']
        print(row['image_size'], row['architecture'], 'AUROC', round(a['mean'], 5), 'SD', round(a['sample_standard_deviation'], 5),
              'Brier', round(b['mean'], 5), 'SD', round(b['sample_standard_deviation'], 5))
    return report


if __name__ == '__main__':
    summarize()
