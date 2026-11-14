"""Independent artifact checks and fresh CPU inference for the frozen November export."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from model import make_model, image_tensor, load_config
from selection import choose, threshold
from research_benchmark import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def saved_scores(path, frame):
    values = pd.read_csv(path).sort_values('sha256').reset_index(drop=True)
    if values.sha256.duplicated().any() or values.isna().any().any():
        raise ValueError('Missing or duplicate prediction records')
    for field in ['sha256', 'patient']:
        if not values[field].equals(frame[field]):
            raise ValueError('Prediction identity mismatch: ' + str(path))
    if not np.array_equal(values.y_true, frame.label):
        raise ValueError('Prediction label mismatch')
    if not np.isfinite(values.probability).all() or not values.probability.between(0, 1).all():
        raise ValueError('Invalid stored probability')
    return values


class Images(Dataset):
    def __init__(self, paths): self.paths = paths
    def __len__(self): return len(self.paths)
    def __getitem__(self, index): return image_tensor(self.paths[index])


def verify(folder, data_root, output, submitted_source_hash=None):
    started = time.monotonic()
    folder, data_root, output = map(Path, [folder, data_root, output])
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    config = load_config(folder)
    summary = json.loads((folder / 'summary.json').read_text())
    config_hash = sha(folder / 'inference_config.json')
    if config_hash != summary['config_sha256']:
        raise ValueError('Frozen configuration hash mismatch')
    source = pd.read_csv(folder / 'split_manifest.csv').sort_values('sha256').reset_index(drop=True)
    if source.groupby('split').size().to_dict() != dict(train=3815, selection=456, calibration=482, test=618):
        raise ValueError('Invalid cohort counts')
    if source.groupby('patient').split.nunique().max() != 1 or source.sha256.duplicated().any():
        raise ValueError('Partition leakage')
    if source.groupby('patient').training_role.nunique().max() != 1:
        raise ValueError('Internal holdout leakage')
    expected_roles = dict(fit=3034, early_stop=781, selection=456, calibration=482, test=618)
    if source.groupby('training_role').size().to_dict() != expected_roles:
        raise ValueError('Unexpected internal cohort sizes')
    for split, expected in [('train', '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d'),
                            ('test', 'c0de6a4a18c8c295d85551b5b8602410e2e26d173b8ab9f141e82cd6fdca9378')]:
        if hashlib.sha256('\n'.join(source.loc[source.split == split, 'sha256']).encode()).hexdigest() != expected:
            raise ValueError('Frozen cohort digest differs')
    frames = {name: source.loc[source.split == name].reset_index(drop=True) for name in ['selection', 'calibration', 'test']}
    records = json.loads((folder / 'development_trials.json').read_text())
    expected_trials = {f'{r}_seed_{s}' for r in ['b0_standard', 'b0_regularized', 'v2s_regularized'] for s in [42, 123, 2026]}
    if len(records) != 9 or {r['trial'] for r in records} != expected_trials:
        raise ValueError('Incomplete planned trial set')
    selection_scores = {}; checkpoint_checks = []
    for rec in records:
        path = folder / rec['trial'] / 'best_weights.pt'
        if sha(path) != rec['checkpoint_sha256']:
            raise ValueError('Checkpoint hash mismatch: ' + rec['trial'])
        model, _ = make_model(rec['architecture'])
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        model.eval()
        with torch.inference_mode():
            score = model(torch.zeros(1, 3, 320, 320))
        if score.shape != (1, 1) or not torch.isfinite(score).all():
            raise ValueError('Checkpoint forward failed')
        del model
        history = pd.read_csv(folder / rec['trial'] / 'history.csv')
        complete = history.loc[history.complete_epoch]
        ranked = complete.assign(correct=np.rint(complete.early_stop_accuracy * 781).astype(int)).sort_values(
            ['correct', 'early_stop_log_loss', 'epoch'], ascending=[False, True, True])
        if int(ranked.iloc[0].epoch) != rec['best_epoch']:
            raise ValueError('Saved checkpoint epoch violates selection rule')
        if len(complete) != rec['epochs_completed']:
            raise ValueError('Completed epoch count mismatch')
        values = saved_scores(folder / rec['trial'] / 'selection_predictions.csv', frames['selection'])
        selection_scores[rec['trial']] = values.probability.to_numpy()
        checkpoint_checks.append(dict(trial=rec['trial'], sha256=rec['checkpoint_sha256'], strict_load_and_forward=True,
                                      best_epoch_rule_verified=True, complete_epochs=len(complete)))
    definitions = {name: [f'{name}_seed_{s}' for s in [42, 123, 2026]] for name in ['b0_standard', 'b0_regularized', 'v2s_regularized']}
    definitions['all_nine'] = [f'{name}_seed_{s}' for name in definitions for s in [42, 123, 2026]]
    choice = choose(frames['selection'].label, frames['selection'].patient, selection_scores, definitions)
    original_choice = json.loads((folder / 'development_choice.json').read_text())
    if choice['selected']['candidate'] != config['candidate'] or choice['selected']['members'] != original_choice['selected']['members']:
        raise ValueError('Development candidate reconstruction differs')
    for candidate, original in zip(choice['candidates'], original_choice['candidates']):
        if candidate['candidate'] != original['candidate'] or candidate['correct'] != original['correct']:
            raise ValueError('Development ranking differs')
    selected_names = [m['trial'] for m in config['members']]
    if selected_names != choice['selected']['members']:
        raise ValueError('Export does not contain development-selected members')
    stored = {}; fresh_results = {}
    for split in ['calibration', 'test']:
        frame = frames[split]
        stored[split] = saved_scores(folder / (split + '_predictions.csv'), frame)
        members = [saved_scores(folder / m['trial'] / (split + '_predictions.csv'), frame).probability.to_numpy()
                   for m in config['members']]
        reconstructed = np.mean(members, axis=0)
        if np.max(np.abs(reconstructed - stored[split].probability.to_numpy())) > 1e-12:
            raise ValueError('Saved ensemble mean mismatch')
        if not np.array_equal(stored[split].prediction, reconstructed >= config['threshold']):
            raise ValueError('Saved classifications violate frozen threshold')
        paths = []
        for row in frame.itertuples():
            path = data_root / row.original_split / ('NORMAL' if row.label == 0 else 'PNEUMONIA') / row.filename
            with Image.open(path) as image:
                gray = image.convert('L')
                digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                if digest != row.sha256: raise ValueError('Decoded image mismatch')
            paths.append(path)
        data = DataLoader(Images(paths), batch_size=12, shuffle=False, num_workers=2)
        member_scores = []
        for member in config['members']:
            model, _ = make_model(member['architecture'])
            model.load_state_dict(torch.load(folder / member['checkpoint'], map_location='cpu', weights_only=True), strict=True)
            model.eval(); p = []
            with torch.inference_mode():
                for batch in data:
                    p.extend(torch.sigmoid(model(batch).flatten()).numpy().astype(float))
            member_scores.append(np.asarray(p))
            del model
            print('CPU_MEMBER_DONE', split, member['trial'], len(p), flush=True)
        fresh = np.mean(member_scores, axis=0)
        errors = np.abs(fresh - stored[split].probability.to_numpy())
        changed = int(np.sum((fresh >= config['threshold']) != stored[split].prediction.to_numpy()))
        frame[['sha256', 'patient', 'label', 'filename']].assign(probability=fresh, prediction=(fresh >= config['threshold']).astype(int)).to_csv(
            output / (split + '_fresh_cpu_predictions.csv'), index=False)
        fresh_results[split] = dict(images=len(frame), pixel_identities_verified=True,
                                   max_absolute_score_difference=float(errors.max()), mean_absolute_score_difference=float(errors.mean()),
                                   changed_predictions=changed, score_tolerance=1e-4,
                                   passed=bool(errors.max() <= 1e-4 and changed == 0),
                                   fresh_metrics=metrics(frame.label, fresh, config['threshold']))
        if split == 'calibration':
            recut, recorrect = threshold(frame.label, fresh)
            fresh_results[split]['fresh_refitted_threshold'] = recut
            fresh_results[split]['fresh_refitted_correct'] = recorrect
            fresh_results[split]['refit_note'] = 'Diagnostic only; frozen model configuration unchanged'
        print('CPU_PARTITION_DONE', split, json.dumps(fresh_results[split]), flush=True)
    original_cut, correct = threshold(frames['calibration'].label, stored['calibration'].probability.to_numpy())
    if abs(original_cut - config['threshold']) > 1e-12 or correct != summary['calibration_in_sample_correct']:
        raise ValueError('Saved calibration cutoff cannot be reconstructed')
    recomputed = metrics(frames['test'].label, stored['test'].probability.to_numpy(), config['threshold'])
    for name, value in summary['test'].items():
        if abs(recomputed[name] - value) > 1e-10:
            raise ValueError('Test summary cannot be recomputed: ' + name)
    result = dict(status='passed' if all(v['passed'] for v in fresh_results.values()) else 'failed',
                  checkpoint_checks=checkpoint_checks, candidate_choice_reproduced=True,
                  calibration_cutoff_reconstructed=True, stored_test_metrics_recomputed=True,
                  config_sha256=config_hash, partitions=fresh_results, elapsed_seconds=time.monotonic() - started,
                  training_source_sha256_lf=submitted_source_hash,
                  environment=dict(python=sys.version, torch=torch.__version__, device='cpu'),
                  scope='Artifact consistency and fresh CPU reproduction on the same cohorts. Does not establish independent patient identity, external accuracy or clinical safety.')
    (output / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('NOVEMBER_CPU_VERIFICATION', json.dumps(result), flush=True)
    if result['status'] != 'passed':
        raise RuntimeError('Strict CPU reproduction failed; retain diagnostic report without relaxing criteria')
