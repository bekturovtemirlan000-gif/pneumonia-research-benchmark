"""Fresh, development-selected training with a separate checkpoint-selection cohort."""
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from research_benchmark import split_cohorts, patient_id, metrics, seed_all, seed_worker
from model import make_model, MEAN, STD, SIZE
from selection import internal_split, choose, threshold
from analyze_benchmark_comparison import analyze

RECIPES = {
    'b0_standard': dict(architecture='efficientnet_b0', mixup_alpha=0., smoothing=0., jitter=0., weight_decay=1e-4),
    'b0_regularized': dict(architecture='efficientnet_b0', mixup_alpha=.2, smoothing=.05, jitter=.15, weight_decay=5e-4),
    'v2s_regularized': dict(architecture='efficientnet_v2_s', mixup_alpha=.2, smoothing=.05, jitter=.15, weight_decay=5e-4),
}
SEEDS = [42, 123, 2026]


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(data_root, previous, cache, out):
    """Reconstruct frozen cohorts and verify pixels without reading old test scores."""
    cache.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(previous / 'split_manifest.csv').sort_values('sha256').reset_index(drop=True)
    fields = ['sha256', 'patient', 'label', 'split', 'original_split', 'filename']
    if source[fields].isna().any().any() or source.sha256.duplicated().any():
        raise ValueError('Malformed source manifest')
    if source.groupby('split').size().to_dict() != dict(train=3815, selection=456, calibration=482, test=618):
        raise ValueError('Cohort counts differ')
    if source.groupby('patient').split.nunique().max() != 1:
        raise ValueError('Group overlap in source manifest')
    paths, cached = [], []
    for i, row in enumerate(source.itertuples()):
        path = data_root / row.original_split / ('NORMAL' if row.label == 0 else 'PNEUMONIA') / row.filename
        if patient_id(row.filename) != row.patient:
            raise ValueError('Group identity changed')
        with Image.open(path) as image:
            gray = image.convert('L')
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
            if digest != row.sha256:
                raise ValueError('Decoded pixel identity mismatch: ' + row.filename)
            destination = cache / (digest + '.png')
            ImageOps.pad(image.convert('RGB'), (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=0).save(destination)
        paths.append(str(path)); cached.append(str(destination))
        if (i + 1) % 500 == 0:
            print('AUDIT', i + 1, len(source), flush=True)
    source['path'], source['cache_path'] = paths, cached
    reconstructed = split_cohorts(source.drop(columns='split'))
    for name, frame in reconstructed.items():
        if set(frame.sha256) != set(source.loc[source.split == name, 'sha256']):
            raise ValueError('Frozen cohort reconstruction changed: ' + name)
    frames = {name: source.loc[source.split == name].reset_index(drop=True) for name in ['train', 'selection', 'calibration', 'test']}
    frames['fit'], frames['early_stop'] = internal_split(frames['train'])
    source['training_role'] = source.sha256.map(dict.fromkeys(frames['fit'].sha256, 'fit') | dict.fromkeys(frames['early_stop'].sha256, 'early_stop')).fillna(source.split)
    source.drop(columns='cache_path').to_csv(out / 'split_manifest.csv', index=False)
    audit = dict(counts={k: dict(images=len(v), groups=int(v.patient.nunique()), normal=int((v.label == 0).sum()), pneumonia=int((v.label == 1).sum())) for k, v in frames.items()},
                 all_pixel_identities_verified=True, original_train_test_digests_verified=True,
                 internal_split='Every fifth SHA256-ordered proxy group within each class; salt 20261115',
                 grouping='Filename-derived proxies, not verified clinical patient identities')
    save(out / 'data_audit.json', audit)
    print('DATA_AUDIT', json.dumps(audit), flush=True)
    return frames


class Radiographs(Dataset):
    def __init__(self, frame, recipe=None):
        self.frame = frame.reset_index(drop=True)
        steps = []
        if recipe is not None:
            steps.append(transforms.RandomAffine(7, translate=(.03, .03), scale=(.95, 1.05), interpolation=transforms.InterpolationMode.BILINEAR))
            if recipe['jitter']:
                steps.append(transforms.ColorJitter(brightness=recipe['jitter'], contrast=recipe['jitter']))
        self.transform = transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(row.cache_path) as image:
            x = self.transform(image.convert('RGB'))
        return x, torch.tensor(row.label, dtype=torch.float32)


def loader(frame, seed=42, recipe=None, workers=2):
    return DataLoader(Radiographs(frame, recipe), batch_size=24, shuffle=recipe is not None,
                      num_workers=workers, pin_memory=True, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def probabilities(model, data, device):
    model.eval(); result = []
    for x, _ in data:
        result.extend(torch.sigmoid(model(x.to(device)).flatten().float()).cpu().numpy().astype(float))
    p = np.asarray(result, dtype=float)
    if len(p) != len(data.dataset) or not np.isfinite(p).all():
        raise ValueError('Invalid inference output')
    return p


def prediction_file(path, frame, p, cut):
    frame[['sha256', 'patient', 'label', 'filename', 'original_split']].assign(
        y_true=frame.label, probability=p, prediction=(p >= cut).astype(int)).to_csv(path, index=False)


def fit_trial(recipe_name, seed, frames, dest, deadline, protocol):
    start = time.monotonic(); seed_all(seed)
    recipe = RECIPES[recipe_name]
    model, head = make_model(recipe['architecture'], pretrained=True)
    model.to('cuda')
    head_ids = {id(p) for p in head.parameters()}
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if id(p) not in head_ids], lr=1e-4),
        dict(params=head.parameters(), lr=1e-3)], weight_decay=recipe['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol['epochs_max'], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')
    loss_fn = nn.BCEWithLogitsLoss()  # Natural class frequency, no inverse-frequency weighting.
    fit_data = loader(frames['fit'], seed, recipe)
    stop_data = loader(frames['early_stop'])
    best_key = None; best_epoch = None; stale = 0; history = []; reason = 'max_epochs'
    for epoch in range(1, protocol['epochs_max'] + 1):
        if time.monotonic() >= deadline:
            reason = 'time_budget'; break
        model.train()
        total = 0.; seen = 0; epoch_start = time.monotonic()
        for x, y in fit_data:
            if time.monotonic() >= deadline:
                reason = 'time_budget'; break
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            y = y * (1 - recipe['smoothing']) + .5 * recipe['smoothing']
            if recipe['mixup_alpha']:
                lam = float(np.random.beta(recipe['mixup_alpha'], recipe['mixup_alpha']))
                order = torch.randperm(len(y), device='cuda')
                x, y = lam * x + (1 - lam) * x[order], lam * y + (1 - lam) * y[order]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = loss_fn(model(x).flatten(), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            scaler.step(optimizer); scaler.update()
            total += float(loss.item()) * len(y); seen += len(y)
        if not seen:
            reason = 'time_budget'; break
        p = probabilities(model, stop_data, 'cuda')
        measured = metrics(frames['early_stop'].label, p, .5)
        complete = seen == len(frames['fit'])
        row = dict(epoch=epoch, train_loss=total / seen, images_seen=seen, complete_epoch=complete,
                   early_stop_accuracy=measured['accuracy'], early_stop_log_loss=measured['log_loss'],
                   early_stop_auroc=measured['auroc'], seconds=time.monotonic() - epoch_start,
                   backbone_lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        pd.DataFrame(history).to_csv(dest / 'history.csv', index=False)
        print('EPOCH', dest.name, json.dumps(row), flush=True)
        # Incomplete epochs never qualify for checkpoint selection.
        if complete:
            key = (int(measured['tn'] + measured['tp']), -measured['log_loss'])
            if best_key is None or key > best_key:
                best_key = key; best_epoch = epoch; stale = 0
                torch.save(model.state_dict(), dest / 'best_weights.pt')
            else:
                stale += 1
        scheduler.step()
        if not complete:
            reason = 'time_budget'; break
        if epoch >= protocol['minimum_epochs'] and stale >= protocol['patience']:
            reason = 'early_stopping'; break
    if best_epoch is None:
        raise RuntimeError('No complete epoch in ' + dest.name)
    model.load_state_dict(torch.load(dest / 'best_weights.pt', map_location='cpu', weights_only=True), strict=True)
    selection = probabilities(model, loader(frames['selection']), 'cuda')
    prediction_file(dest / 'selection_predictions.csv', frames['selection'], selection, .5)
    record = dict(trial=dest.name, recipe=recipe_name, architecture=recipe['architecture'], seed=seed,
                  best_epoch=best_epoch, epochs_completed=sum(r['complete_epoch'] for r in history),
                  stop_reason=reason, seconds=time.monotonic() - start,
                  best_internal_correct=best_key[0], best_internal_log_loss=-best_key[1],
                  selection_at_0_5=metrics(frames['selection'].label, selection, .5),
                  checkpoint_sha256=sha(dest / 'best_weights.pt'), parameters=sum(p.numel() for p in model.parameters()))
    save(dest / 'development_summary.json', record)
    del model, head, optimizer, scheduler, scaler, fit_data, stop_data
    gc.collect(); torch.cuda.empty_cache()
    print('TRIAL_DONE', json.dumps(record), flush=True)
    return record, selection


def run(data_root, previous, out, protocol, previous_lock):
    start = time.monotonic()
    data_root, previous, out = map(Path, [data_root, previous, out])
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'protocol.json', protocol)
    old_config = json.loads((previous / 'inference_config.json').read_text())
    for key, expected in previous_lock['config'].items():
        if old_config.get(key) != expected:
            raise ValueError('Previous model configuration changed: ' + key)
    for trial, expected in previous_lock['checkpoints'].items():
        if sha(previous / trial / 'best_weights.pt') != expected:
            raise ValueError('Previous checkpoint changed: ' + trial)
    save(out / 'previous_lock.json', previous_lock)
    print('PREVIOUS_MODEL_LOCK_PASSED', len(previous_lock['checkpoints']), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('Enable Kaggle GPU')
    torch.set_num_threads(2)
    seed_all(42)
    env = dict(python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
               gpu=torch.cuda.get_device_name(0), visible_gpus=torch.cuda.device_count(), used_gpus=1)
    save(out / 'environment.json', env)
    print('ENVIRONMENT', json.dumps(env), flush=True)
    frames = prepare(data_root, previous, Path('/kaggle/temp/november_cache'), out)
    names = list(RECIPES)
    trials = [(names[(i + j) % len(names)], seed) for i, seed in enumerate(SEEDS) for j in range(len(names))]
    training_start = time.monotonic(); deadline = training_start + protocol['training_budget_minutes'] * 60
    records, scores = [], {}
    for i, (recipe, seed) in enumerate(trials):
        remaining = deadline - time.monotonic()
        if remaining < 120:
            raise RuntimeError('Insufficient budget for all planned trials; retain partial outputs')
        trial = f'{recipe}_seed_{seed}'
        dest = out / trial; dest.mkdir(exist_ok=True)
        record, scores[trial] = fit_trial(recipe, seed, frames, dest,
                                          time.monotonic() + remaining / (len(trials) - i), protocol)
        records.append(record)
        save(out / 'development_trials.json', records)
    training_seconds = time.monotonic() - training_start
    definitions = {name: [f'{name}_seed_{seed}' for seed in SEEDS] for name in RECIPES}
    definitions['all_nine'] = [f'{name}_seed_{seed}' for name in RECIPES for seed in SEEDS]
    choice = choose(frames['selection'].label, frames['selection'].patient, scores, definitions)
    save(out / 'development_choice.json', choice)
    print('DEVELOPMENT_CHOICE', json.dumps(choice), flush=True)
    selected = choice['selected']['members']
    by_id = {r['trial']: r for r in records}
    members = [dict(trial=t, architecture=by_id[t]['architecture'], checkpoint=t + '/best_weights.pt',
                    sha256=by_id[t]['checkpoint_sha256']) for t in selected]
    calibration_members = []
    for member in members:
        model, _ = make_model(member['architecture'])
        model.load_state_dict(torch.load(out / member['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.to('cuda')
        p = probabilities(model, loader(frames['calibration']), 'cuda')
        calibration_members.append(p)
        prediction_file(out / member['trial'] / 'calibration_predictions.csv', frames['calibration'], p, .5)
        del model; torch.cuda.empty_cache()
    pc = np.mean(calibration_members, axis=0)
    cutoff, cal_correct = threshold(frames['calibration'].label, pc)
    config = dict(image_size=SIZE, members=members, aggregation='mean_raw_sigmoid', threshold=cutoff,
                  classes=['NORMAL', 'PNEUMONIA'], candidate=choice['selected']['candidate'],
                  preprocessing='RGB; bicubic full-image letterbox 320; ToTensor; ImageNet normalization',
                  probability_calibration='None; raw scores are not clinical probabilities',
                  research_status='Exploratory; source cohorts previously inspected; no external clinical validation')
    save(out / 'inference_config.json', config)
    for filename in ['model.py', 'selection.py', 'train_experiment.py', 'research_benchmark.py', 'compare_experiments.py', 'analyze_benchmark_comparison.py']:
        shutil.copyfile(Path(__file__).parent / filename, out / filename)
    config_hash = sha(out / 'inference_config.json')
    print('MODEL_FROZEN', json.dumps(dict(config=config, config_sha256=config_hash)), flush=True)
    prediction_file(out / 'calibration_predictions.csv', frames['calibration'], pc, cutoff)
    ps = np.mean([scores[t] for t in selected], axis=0)
    prediction_file(out / 'selection_predictions.csv', frames['selection'], ps, cutoff)
    # No test score inference or previous test prediction access occurs above this boundary.
    test_members = []
    for member in members:
        if sha(out / member['checkpoint']) != member['sha256']:
            raise ValueError('Weights changed after freeze')
        model, _ = make_model(member['architecture'])
        model.load_state_dict(torch.load(out / member['checkpoint'], map_location='cpu', weights_only=True), strict=True)
        model.to('cuda')
        p = probabilities(model, loader(frames['test']), 'cuda')
        test_members.append(p)
        prediction_file(out / member['trial'] / 'test_predictions.csv', frames['test'], p, .5)
        del model; torch.cuda.empty_cache()
    pt = np.mean(test_members, axis=0)
    prediction_file(out / 'test_predictions.csv', frames['test'], pt, cutoff)
    comparison = analyze(previous, out, out / 'paired_comparison', replicates=1000)
    # Exercise the actual exported inference CLI on two hash-selected images.
    cli = []
    for index in np.argsort(frames['test'].sha256.to_numpy())[:2]:
        row = frames['test'].iloc[int(index)]
        execution = subprocess.run([sys.executable, str(out / 'model.py'), '--model-dir', str(out),
                                    '--image', row.path, '--device', 'cpu'], capture_output=True, text=True, check=True)
        observed = json.loads(execution.stdout.strip().splitlines()[-1])
        error = abs(observed['score'] - float(pt[index]))
        changed = observed['prediction'] != int(pt[index] >= cutoff)
        cli.append(dict(sha256=row.sha256, score_difference=error, changed_prediction=changed,
                        passed=error <= 1e-4 and not changed))
    result = dict(status='completed', selected_candidate=choice['selected']['candidate'],
                  development_choice=choice['selected'], calibration_in_sample_correct=cal_correct,
                  calibration_images=len(pc), threshold=cutoff, test=metrics(frames['test'].label, pt, cutoff),
                  test_at_0_5=metrics(frames['test'].label, pt, .5), training_seconds=training_seconds,
                  total_seconds=time.monotonic() - start, trials=records, config_sha256=config_hash,
                  cli_smoke=cli, cli_all_passed=all(r['passed'] for r in cli), environment=env,
                  test_status='Previously inspected source test; exploratory only',
                  uncertainty=comparison)
    if sha(out / 'inference_config.json') != config_hash:
        raise ValueError('Configuration changed after freeze')
    save(out / 'summary.json', result)
    print('TRAINING_EXPERIMENT_DONE', json.dumps(result), flush=True)
    if not result['cli_all_passed']:
        raise RuntimeError('Export CLI verification failed; preserve results and investigate')
