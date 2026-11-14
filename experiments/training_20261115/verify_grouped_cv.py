"""Reconstruct all CV artifacts and independently repeat outer-fold inference on CPU."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from model import make_model
from selection import internal_split, threshold
from verify_model import Images, saved_scores, sha
from research_benchmark import metrics
from compare_experiments import metrics as frame_metrics
from cv_geometry_control import run as run_geometry_control


def verify(folder, data_root, output, expected_summary_sha256):
    start = time.monotonic()
    folder, data_root, output = map(Path, [folder, data_root, output])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if sha(folder / 'summary.json') != expected_summary_sha256:
        raise ValueError('Unexpected frozen CV summary')
    result = json.loads((folder / 'summary.json').read_text())
    frame = pd.read_csv(folder / 'outer_manifest.csv').sort_values('sha256').reset_index(drop=True)
    if len(frame) != 3815 or frame.sha256.duplicated().any() or frame.patient.nunique() != 2153:
        raise ValueError('Invalid source CV manifest')
    if hashlib.sha256('\n'.join(frame.sha256).encode()).hexdigest() != '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d':
        raise ValueError('Original training cohort digest mismatch')
    fold_indices = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261115)
                        .split(np.zeros(len(frame)), frame.label, frame.patient))
    expected_ids = np.full(len(frame), -1, dtype=int)
    for fold, (_, held) in enumerate(fold_indices): expected_ids[held] = fold
    if not np.array_equal(expected_ids, frame.outer_fold): raise ValueError('Outer folds do not reconstruct')
    recipes = ['b0_standard', 'b0_regularized', 'v2s_regularized']
    seeds = [42, 123, 2026]
    expected_trials = {f'{recipe}_fold_{fold}_seed_{seed}' for fold in range(3) for recipe in recipes for seed in seeds}
    records = result['training_records']
    if len(records) != 27 or {r['trial'] for r in records} != expected_trials:
        raise ValueError('Expected all 27 fresh models')
    if result['primary_model_changed'] or result['original_selection_calibration_test_scored']:
        raise ValueError('CV scope changed')
    paths = {}; dimensions = []
    for row in frame.itertuples():
        path = data_root / row.original_split / ('NORMAL' if row.label == 0 else 'PNEUMONIA') / row.filename
        with Image.open(path) as image:
            dimensions.append(image.size)
            gray = image.convert('L')
            if hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest() != row.sha256:
                raise ValueError('Decoded CV image identity changed')
        paths[row.sha256] = path
    frame['width'], frame['height'] = np.asarray(dimensions).T
    geometry, geometry_scores = run_geometry_control(frame, fold_indices, output / 'geometry_control')
    checkpoint_checks = []
    for rec in records:
        trial = folder / rec['trial']
        if sha(trial / 'best_weights.pt') != rec['checkpoint_sha256']:
            raise ValueError('CV checkpoint hash changed')
        outer_train, outer_test = fold_indices[rec['fold']]
        fit, stop = internal_split(frame.iloc[outer_train].reset_index(drop=True))
        held = frame.iloc[outer_test].reset_index(drop=True)
        manifest = pd.read_csv(trial / 'fold_manifest.csv')
        for role, cohort in [('fit', fit), ('checkpoint_and_cutoff', stop), ('outer_test', held)]:
            if set(manifest.loc[manifest.role == role, 'sha256']) != set(cohort.sha256):
                raise ValueError('CV role differs from prescribed split')
        if manifest.groupby('patient').role.nunique().max() != 1:
            raise ValueError('CV proxy group leakage')
        h = pd.read_csv(trial / 'history.csv')
        full = h.loc[h.complete_epoch]
        ranked = full.assign(correct=np.rint(full.early_stop_accuracy * len(stop)).astype(int)).sort_values(
            ['correct', 'early_stop_log_loss', 'epoch'], ascending=[False, True, True])
        if int(ranked.iloc[0].epoch) != rec['best_epoch'] or len(full) != rec['epochs_completed']:
            raise ValueError('Checkpoint selection cannot be reconstructed')
        checkpoint_checks.append(dict(trial=rec['trial'], sha256=rec['checkpoint_sha256'], split_and_epoch_verified=True))
    fold_reports = []; pooled_fresh = {recipe: [] for recipe in recipes}
    for fold in range(3):
      for recipe in recipes:
        outer_train, outer_test = fold_indices[fold]
        _, stop = internal_split(frame.iloc[outer_train].reset_index(drop=True))
        held = frame.iloc[outer_test].reset_index(drop=True)
        directory = folder / f'{recipe}_fold_{fold}_ensemble'
        config = json.loads((directory / 'frozen_fold_config.json').read_text())
        reported = next(r for r in result['folds'] if r['fold'] == fold and r['recipe'] == recipe)
        if sha(directory / 'frozen_fold_config.json') != reported['frozen_config_sha256']:
            raise ValueError('Frozen fold configuration changed')
        expected_members = [f'{recipe}_fold_{fold}_seed_{seed}' for seed in seeds]
        if [r['trial'] for r in config['members']] != expected_members:
            raise ValueError('Fold ensemble membership changed')
        inner = [saved_scores(folder / name / 'checkpoint_holdout_predictions.csv', stop).probability.to_numpy() for name in expected_members]
        cut, correct = threshold(stop.label, np.mean(inner, axis=0))
        if abs(cut - config['threshold']) > 1e-12 or correct != reported['internal_fit_correct']:
            raise ValueError('Fold cutoff cannot be reconstructed from inner data')
        stored = saved_scores(directory / 'outer_predictions.csv', held)
        member_outer = [saved_scores(folder / name / 'outer_predictions.csv', held).probability.to_numpy() for name in expected_members]
        raw = np.mean(member_outer, axis=0)
        if np.max(np.abs(raw - stored.probability.to_numpy())) > 1e-12:
            raise ValueError('Outer ensemble cannot be reconstructed')
        if not np.array_equal(stored.prediction.to_numpy(), raw >= cut):
            raise ValueError('Outer predictions use wrong cutoff')
        recomputed = metrics(held.label, raw, cut)
        for key, value in reported['outer_metrics'].items():
            if abs(recomputed[key] - value) > 1e-10:
                raise ValueError('Fold metrics differ')
        data = DataLoader(Images([paths[h] for h in held.sha256]), batch_size=12, shuffle=False, num_workers=2)
        fresh_members = []
        for member in config['members']:
            if sha(folder / member['checkpoint']) != member['sha256']: raise ValueError('Fold member hash changed')
            model, _ = make_model(config['architecture'])
            model.load_state_dict(torch.load(folder / member['checkpoint'], map_location='cpu', weights_only=True), strict=True)
            model.eval(); values = []
            with torch.inference_mode():
                for batch in data: values.extend(torch.sigmoid(model(batch).flatten()).numpy().astype(float))
            fresh_members.append(np.asarray(values))
            del model
            print('CV_CPU_MEMBER', member['trial'], len(values), flush=True)
        fresh = np.mean(fresh_members, axis=0)
        error = np.abs(fresh - raw)
        changed = int(np.sum((fresh >= cut) != stored.prediction.to_numpy()))
        scores = held[['sha256', 'patient', 'label']].assign(y_true=held.label, probability=fresh,
                     prediction=(fresh >= cut).astype(int), fold=fold, threshold=cut)
        scores.to_csv(output / f'{recipe}_fold_{fold}_cpu_predictions.csv', index=False)
        pooled_fresh[recipe].append(scores)
        report = dict(recipe=recipe, fold=fold, images=len(held), max_absolute_score_difference=float(error.max()),
                      changed_predictions=changed, score_tolerance=1e-4, passed=bool(error.max() <= 1e-4 and changed == 0),
                      stored_metrics_recomputed=True, cutoff_reconstructed=True, members=len(fresh_members))
        fold_reports.append(report)
        print('CV_CPU_FOLD_DONE', json.dumps(report), flush=True)
    pooled_checks = {}
    for recipe in recipes:
        stored = saved_scores(folder / (recipe + '_oof_predictions.csv'), frame)
        reported = result['recipes'][recipe]
        if int(np.sum(stored.y_true == stored.prediction)) != reported['correct']:
            raise ValueError('Pooled correct count mismatch')
        for key, value in frame_metrics(stored).items():
            if abs(value - reported['pooled_metrics'][key]) > 1e-10:
                raise ValueError('Pooled metric mismatch')
        fresh = pd.concat(pooled_fresh[recipe]).sort_values('sha256').reset_index(drop=True)
        if not fresh.sha256.equals(frame.sha256): raise ValueError('Fresh OOF coverage incomplete')
        fresh.to_csv(output / (recipe + '_oof_cpu_predictions.csv'), index=False)
        pooled_checks[recipe] = dict(images=len(fresh), groups=int(fresh.patient.nunique()),
                                    fresh_metrics=frame_metrics(fresh),
                                    changed_predictions=int(np.sum(fresh.prediction != stored.prediction)))
        if not stored.sha256.equals(geometry_scores.sha256): raise ValueError('Control pairing mismatch')
        pooled_checks[recipe]['geometry_paired_correctness'] = dict(
            neural_only_correct=int(np.sum((stored.prediction == stored.y_true) & (geometry_scores.prediction != stored.y_true))),
            geometry_only_correct=int(np.sum((stored.prediction != stored.y_true) & (geometry_scores.prediction == stored.y_true))),
            both_correct=int(np.sum((stored.prediction == stored.y_true) & (geometry_scores.prediction == stored.y_true))),
            neither_correct=int(np.sum((stored.prediction != stored.y_true) & (geometry_scores.prediction != stored.y_true))),
            accuracy_difference=float(np.mean(stored.prediction == stored.y_true) - geometry['pooled_metrics']['accuracy']))
    report = dict(status='passed' if all(r['passed'] for r in fold_reports) else 'failed',
                  verified_checkpoints=27, image_identities_verified=3815, proxy_groups=2153,
                  folds_reconstructed=True, checkpoint_checks=checkpoint_checks, fold_reports=fold_reports,
                  pooled=pooled_checks, geometry_control=geometry, elapsed_seconds=time.monotonic() - start,
                  summary_sha256=expected_summary_sha256, device='cpu', torch=torch.__version__,
                  scope='Same-data CPU reproduction and artifact integrity. Not new external evidence or a new selection rule.')
    (output / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('GROUPED_CV_CPU_VERIFICATION', json.dumps(report), flush=True)
    if report['status'] != 'passed': raise RuntimeError('Strict grouped-CV reproduction failed; report retained')
