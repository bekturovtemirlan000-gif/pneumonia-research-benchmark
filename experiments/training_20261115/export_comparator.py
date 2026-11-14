"""Export the already-frozen original comparator scores; no fitting or inference."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_experiments import metrics
from export_predictions import export


def run(folder, output, lock, expected_metrics):
    folder, output = Path(folder), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = folder / 'inference_config.json'
    config = json.loads(config_path.read_text())
    for key, value in lock['config'].items():
        if config.get(key) != value:
            raise ValueError('Original comparator configuration differs: ' + key)
    for trial, expected in lock['checkpoints'].items():
        if hashlib.sha256((folder/trial/'best_weights.pt').read_bytes()).hexdigest() != expected:
            raise ValueError('Original checkpoint changed: ' + trial)
    manifest = pd.read_csv(folder/'split_manifest.csv').sort_values('sha256').reset_index(drop=True)
    cohort = manifest.loc[manifest.split == 'test'].reset_index(drop=True)
    if len(cohort) != 618 or cohort.patient.nunique() != 426 or cohort.sha256.duplicated().any():
        raise ValueError('Original test cohort invalid')
    digest = hashlib.sha256('\n'.join(cohort.sha256).encode()).hexdigest()
    if digest != 'c0de6a4a18c8c295d85551b5b8602410e2e26d173b8ab9f141e82cd6fdca9378':
        raise ValueError('Original test cohort digest differs')
    source = folder/'test_predictions.csv'
    values = pd.read_csv(source).sort_values('sha256').reset_index(drop=True)
    if values.isna().any().any() or values.sha256.duplicated().any():
        raise ValueError('Invalid original score table')
    for key in ['sha256', 'patient']:
        if not cohort[key].equals(values[key]): raise ValueError('Original score identity mismatch')
    if not np.array_equal(values.y_true, cohort.label): raise ValueError('Original labels differ')
    if not np.isfinite(values.probability).all() or not values.probability.between(0, 1).all():
        raise ValueError('Invalid original score')
    if not np.array_equal(values.prediction, values.probability >= config['threshold']):
        raise ValueError('Original classifications differ from frozen rule')
    measured = metrics(values)
    for key, expected in expected_metrics.items():
        if abs(measured[key]-expected) > 1e-10: raise ValueError('Original summary mismatch: ' + key)
    public = values[['sha256', 'patient', 'y_true', 'probability', 'prediction']].copy()
    public['threshold'] = config['threshold']
    meta = export(public, output, 'public_original_320_test_predictions.csv')
    result = dict(status='passed', original_kaggle_version=360888746, images=618, groups=426,
                  checkpoint_hashes_verified=len(lock['checkpoints']), frozen_configuration=lock['config'],
                  config_file_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  source_prediction_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  metrics=measured, public_prediction_exports={'original_320': meta},
                  scope='Artifact export after outcomes were known. No fitting, inference, cutoff change or model selection. Scores are the saved original temperature-scaled scores, not raw sigmoid probabilities.')
    (output/'comparator_export.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('COMPARATOR_EXPORT_DONE', json.dumps(result), flush=True)
    return result
