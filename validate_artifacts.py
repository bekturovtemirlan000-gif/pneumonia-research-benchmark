"""Validate saved experiment consistency without loading PyTorch or images."""
import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from predict import load_config


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as file:
        return list(csv.DictReader(file))


def validate(directory):
    directory = Path(directory)
    metrics = json.loads((directory / 'metrics.json').read_text())
    manifest = read_csv(directory / 'split_manifest.csv')
    split_names = {r['split'] for r in manifest}
    if split_names not in [{'train', 'validation', 'test'}, {'train', 'selection', 'calibration', 'test'}]:
        raise ValueError('Unexpected or missing manifest split')
    for required in ({'test', 'validation'} if 'validation' in split_names else {'test'}):
        if required not in metrics:
            raise ValueError('Missing reported metrics: ' + required)
    for field in ['path', 'sha256']:
        if len({r[field] for r in manifest}) != len(manifest):
            raise ValueError(f'Duplicate manifest {field}')
    if any(not r[field] for r in manifest for field in ['path', 'patient', 'sha256']):
        raise ValueError('Missing manifest identity field')
    partitions = {name: [r for r in manifest if r['split'] == name]
                  for name in sorted(split_names)}
    for name, rows in partitions.items():
        if not rows or {r['label'] for r in rows} != {'0', '1'}:
            raise ValueError(f'{name} must have both classes')
    for field in ['patient', 'sha256']:
        sets = {name: {r[field] for r in rows} for name, rows in partitions.items()}
        for a, b in combinations(partitions, 2):
            if sets[a] & sets[b]:
                raise ValueError(f'{field} overlap between {a} and {b}')
    threshold = metrics['threshold']
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold outside probability range')
    confusion = {}
    for name in sorted(split_names - {'train'}):
        rows = read_csv(directory / f'{name}_predictions.csv')
        expected = {(r['path'], r['label'], r['patient'], r['sha256']) for r in partitions[name]}
        observed = {(r['path'], r['y_true'], r['patient'], r['sha256']) for r in rows}
        if expected != observed or len(rows) != len(partitions[name]):
            raise ValueError(f'{name} predictions do not match the split manifest')
        tn = fp = fn = tp = 0
        for row in rows:
            score = float(row['probability'])
            y, pred = int(row['y_true']), int(row['prediction'])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Non-finite or invalid prediction score')
            if pred != int(score >= threshold):
                raise ValueError('Prediction disagrees with saved threshold')
            tn += y == 0 and pred == 0
            fp += y == 0 and pred == 1
            fn += y == 1 and pred == 0
            tp += y == 1 and pred == 1
        calculated = {'accuracy': (tn+tp)/len(rows), 'sensitivity': tp/(tp+fn),
                      'specificity': tn/(tn+fp)}
        if name in metrics:
            for metric, value in calculated.items():
                if not math.isclose(value, metrics[name][metric], abs_tol=1e-8):
                    raise ValueError(f'{name} {metric} does not match predictions')
        confusion[name] = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    if (directory / 'inference_config.json').exists():
        config, paths = load_config(directory)
        if config['threshold'] != threshold:
            raise ValueError('Inference and evaluation thresholds disagree')
        if config.get('temperature', 1) != metrics.get('temperature', 1):
            raise ValueError('Inference and evaluation temperatures disagree')
        checkpoints = [str(p.relative_to(directory.resolve())) for p in paths]
    else:
        checkpoints = ['pneumonia_model.pt', 'best_weights.pt']
        if not all((directory / name).is_file() for name in checkpoints):
            raise ValueError('Baseline weights missing')
    return {'validated': True, 'counts': {k: len(v) for k, v in partitions.items()},
            'confusion_matrices': confusion, 'checkpoints': checkpoints}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.directory), indent=2))
