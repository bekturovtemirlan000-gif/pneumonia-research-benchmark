"""Secondary three-fold grouped assessment on the original training cohort only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
import torch
import torchvision
from model import make_model, SIZE
from selection import internal_split, threshold
from train_experiment import RECIPES, fit_trial, loader, probabilities, prediction_file, sha, save
from research_benchmark import patient_id, metrics
from compare_experiments import metrics as frame_metrics


def run(data_root, previous, out, protocol, previous_lock):
    started = time.monotonic()
    data_root, previous, out = map(Path, [data_root, previous, out])
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', protocol)
    if not torch.cuda.is_available(): raise RuntimeError('Enable Kaggle GPU')
    torch.set_num_threads(2)
    for trial, expected in previous_lock['checkpoints'].items():
        if sha(previous / trial / 'best_weights.pt') != expected:
            raise ValueError('Previous checkpoint lock mismatch')
    source = pd.read_csv(previous / 'split_manifest.csv')
    source = source.loc[source.split == 'train'].sort_values('sha256').reset_index(drop=True)
    if len(source) != 3815 or source.sha256.duplicated().any() or source.groupby('patient').label.nunique().max() != 1:
        raise ValueError('Invalid source training cohort')
    cohort_hash = hashlib.sha256('\n'.join(source.sha256).encode()).hexdigest()
    if cohort_hash != '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d':
        raise ValueError('Frozen original training cohort changed')
    cache = Path('/kaggle/temp/november_cv_cache'); cache.mkdir(parents=True, exist_ok=True)
    paths, caches = [], []
    for index, row in enumerate(source.itertuples()):
        path = data_root / row.original_split / ('NORMAL' if row.label == 0 else 'PNEUMONIA') / row.filename
        if patient_id(row.filename) != row.patient: raise ValueError('Filename group changed')
        with Image.open(path) as image:
            gray = image.convert('L')
            if hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest() != row.sha256:
                raise ValueError('Pixel identity changed')
            cached = cache / (row.sha256 + '.png')
            ImageOps.pad(image.convert('RGB'), (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=0).save(cached)
        paths.append(str(path)); caches.append(str(cached))
        if (index + 1) % 500 == 0: print('CV_AUDIT', index + 1, len(source), flush=True)
    source['path'], source['cache_path'] = paths, caches
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261115)
                 .split(np.zeros(len(source)), source.label, source.patient))
    fold_ids = np.full(len(source), -1, dtype=int)
    for fold, (train, test) in enumerate(folds):
        if set(source.iloc[train].patient) & set(source.iloc[test].patient): raise ValueError('Outer group leakage')
        if set(source.iloc[test].label) != {0, 1}: raise ValueError('Both classes required per fold')
        fold_ids[test] = fold
    source['outer_fold'] = fold_ids
    source.drop(columns='cache_path').to_csv(out / 'outer_manifest.csv', index=False)
    environment = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
                       numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                       gpu=torch.cuda.get_device_name(0), used_gpus=1)
    save(out / 'environment.json', environment)
    print('CV_ENVIRONMENT', json.dumps(environment), flush=True)
    print('CV_FOLDS', json.dumps({str(f): dict(images=int(np.sum(fold_ids == f)), groups=int(source.loc[source.outer_fold == f].patient.nunique()),
                                              normal=int(np.sum((fold_ids == f) & (source.label == 0))),
                                              pneumonia=int(np.sum((fold_ids == f) & (source.label == 1)))) for f in range(3)}), flush=True)
    names = list(RECIPES)
    trials = [(fold, names[(fold + j) % 3]) for fold in range(3) for j in range(3)]
    train_start = time.monotonic(); deadline = train_start + protocol['training_budget_minutes'] * 60
    records = []; all_predictions = {name: [] for name in RECIPES}
    for index, (fold, recipe) in enumerate(trials):
        outer_train, outer_test = folds[fold]
        fit, stop = internal_split(source.iloc[outer_train].reset_index(drop=True))
        held = source.iloc[outer_test].reset_index(drop=True)
        if set(held.patient) & (set(fit.patient) | set(stop.patient)):
            raise ValueError('Inner/outer leakage')
        dest = out / f'{recipe}_fold_{fold}'; dest.mkdir(exist_ok=True)
        pd.concat([fit.assign(role='fit'), stop.assign(role='checkpoint_and_cutoff'), held.assign(role='outer_test')]).drop(columns='cache_path').to_csv(
            dest / 'fold_manifest.csv', index=False)
        remaining = deadline - time.monotonic()
        if remaining < 120: raise RuntimeError('Insufficient CV budget; preserve partial outputs')
        # Pass the inner holdout for both helper evaluation roles; outer images are not scored here.
        rec, inner_p = fit_trial(recipe, 42, dict(fit=fit, early_stop=stop, selection=stop), dest,
                                time.monotonic() + remaining / (len(trials) - index), protocol)
        rec['internal_checkpoint_holdout_at_0_5'] = rec.pop('selection_at_0_5')
        save(dest / 'development_summary.json', rec)
        (dest / 'selection_predictions.csv').rename(dest / 'checkpoint_holdout_predictions.csv')
        cut, count = threshold(stop.label, inner_p)
        frozen = dict(recipe=recipe, fold=fold, threshold=cut, architecture=RECIPES[recipe]['architecture'],
                      checkpoint_sha256=sha(dest / 'best_weights.pt'), primary_model_changed=False)
        save(dest / 'frozen_fold_config.json', frozen)
        print('CV_FOLD_FROZEN', json.dumps(frozen), flush=True)
        model, _ = make_model(RECIPES[recipe]['architecture'])
        model.load_state_dict(torch.load(dest / 'best_weights.pt', map_location='cpu', weights_only=True), strict=True)
        model.to('cuda')
        outer_p = probabilities(model, loader(held), 'cuda')
        del model; torch.cuda.empty_cache()
        prediction_file(dest / 'inner_predictions.csv', stop, inner_p, cut)
        prediction_file(dest / 'outer_predictions.csv', held, outer_p, cut)
        # Outer results cannot change weights, cutoff, remaining recipe settings, or the primary experiment.
        fold_record = dict(recipe=recipe, fold=fold, seed=42, fit_images=len(fit), internal_images=len(stop), outer_images=len(held),
                           threshold=cut, internal_fit_correct=count, outer_metrics=metrics(held.label, outer_p, cut),
                           training=rec, checkpoint_sha256=sha(dest / 'best_weights.pt'))
        records.append(fold_record); save(out / 'fold_results.json', records)
        all_predictions[recipe].append(held[['sha256', 'patient', 'label']].assign(
            y_true=held.label, probability=outer_p, prediction=(outer_p >= cut).astype(int), fold=fold, threshold=cut))
        print('CV_FOLD_DONE', json.dumps(fold_record), flush=True)
    training_seconds = time.monotonic() - train_start
    reports = {}
    for recipe, pieces in all_predictions.items():
        if len(pieces) != 3: raise ValueError('Incomplete three-fold recipe')
        values = pd.concat(pieces).sort_values('sha256').reset_index(drop=True)
        if not values.sha256.equals(source.sha256) or values.sha256.duplicated().any(): raise ValueError('OOF identities differ')
        values.to_csv(out / (recipe + '_oof_predictions.csv'), index=False)
        reports[recipe] = dict(pooled_metrics=frame_metrics(values), images=len(values), groups=int(values.patient.nunique()),
                               correct=int(np.sum(values.y_true == values.prediction)),
                               fold_accuracies=[r['outer_metrics']['accuracy'] for r in records if r['recipe'] == recipe],
                               interpretation='Each image is predicted by a model that excluded its proxy group from parameter fitting, checkpoint choice and cutoff fitting; pooled AUROC combines differently scaled fold models.')
    result = dict(status='completed', images=3815, filename_proxy_groups=int(source.patient.nunique()),
                  recipes=reports, folds=records, training_seconds=training_seconds,
                  total_seconds=time.monotonic() - started, environment=environment,
                  primary_model_changed=False, original_selection_calibration_test_scored=False,
                  limitations=['Secondary grouped internal assessment on the original training cohort; not external testing',
                               'One initialization per fold; fold training sets overlap and are not independent repetitions',
                               'Filename groups are unverified patient proxies; near-duplicate independence remains unresolved',
                               'Checkpoint and cutoff share an inner holdout, but outer-fold groups are excluded from both',
                               'Recipes are fixed in advance; results do not change the primary experiment or choose a deployed model'])
    save(out / 'summary.json', result)
    for name in ['model.py', 'selection.py', 'train_experiment.py', 'run_grouped_cv.py', 'research_benchmark.py', 'compare_experiments.py', 'analyze_benchmark_comparison.py']:
        (out / name).write_text((Path(__file__).parent / name).read_text(encoding='utf-8'), encoding='utf-8')
    print('GROUPED_CV_DONE', json.dumps(result), flush=True)
