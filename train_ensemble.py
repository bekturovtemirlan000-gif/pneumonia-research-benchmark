"""Kaggle pneumonia classification baseline. Research use only.

Run in Kaggle with paultimothymooney/chest-xray-pneumonia attached,
a GPU accelerator, and Internet enabled for ImageNet weights.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import random
import re
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 8
RUN_STARTED = time.monotonic()
TRAIN_DEADLINE = RUN_STARTED + 110 * 60  # Reserve about 10 min for evaluation/export.
OUT = Path('/kaggle/working/pneumonia_ensemble_outputs')
OUT.mkdir(parents=True, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Enable a GPU accelerator in Kaggle Settings, then rerun.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)

# Filename IDs are proxies, not independently verified clinical patient IDs.
def patient_id(name):
    patterns = [(r'person(\d+)_', 'person'),
                (r'NORMAL2-IM-(\d+)-', 'normal2'),
                (r'IM-(\d+)-', 'im')]
    for pattern, prefix in patterns:
        match = re.match(pattern, name, flags=re.I)
        if match:
            return prefix + ':' + match.group(1)
    raise ValueError('Unknown filename pattern; review grouping before training: ' + name)

roots = sorted({p.parent for p in Path('/kaggle/input').rglob('train')
                if (p / 'NORMAL').is_dir() and (p.parent / 'test' / 'PNEUMONIA').is_dir()},
               key=lambda p: (len(p.parts), str(p)))
if not roots:
    raise RuntimeError('Attach the Chest X-Ray Images (Pneumonia) dataset first.')
ROOT = roots[0]  # Select one copy; some dataset versions have nested duplicates.
print('Dataset root:', ROOT, flush=True)
rows = []
for split in ['train', 'val', 'test']:
    for label, cls in enumerate(['NORMAL', 'PNEUMONIA']):
        for path in sorted((ROOT / split / cls).glob('*')):
            if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            with Image.open(path) as im:
                pixels = im.convert('L')
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            rows.append(dict(path=str(path), filename=path.name, label=label,
                             original_split=split, patient=patient_id(path.name), sha256=digest))
df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError('No images found.')
if df.groupby('sha256').label.nunique().max() > 1:
    raise RuntimeError('Identical decoded images have conflicting labels; review dataset.')
# Preserve the provided test partition. Remove overlapping development subjects
# and exact pixel duplicates without inspecting predictions or tuning on test.
test = df[df.original_split == 'test'].drop_duplicates('sha256').copy()
dev0 = df[df.original_split != 'test'].copy()
dev = dev0[~dev0.patient.isin(test.patient) & ~dev0.sha256.isin(test.sha256)]
dev = dev.drop_duplicates('sha256').reset_index(drop=True)
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
train_idx, val_idx = next(splitter.split(dev, dev.label, dev.patient))
train, val = dev.iloc[train_idx].copy(), dev.iloc[val_idx].copy()
for a, b in [(train, val), (train, test), (val, test)]:
    assert not set(a.patient) & set(b.patient)
    assert not set(a.sha256) & set(b.sha256)
for frame in [train, val, test]:
    assert set(frame.label) == {0, 1}
manifest = pd.concat([frame.assign(split=name) for name, frame in
                      [('train', train), ('validation', val), ('test', test)]])
manifest.to_csv(OUT / 'split_manifest.csv', index=False)
audit = {'raw_images': len(df), 'development_removed': len(dev0)-len(dev),
         'test_duplicates_removed': int((df.original_split == 'test').sum()-len(test)),
         'grouping': 'filename-derived patient proxies; not verified clinical IDs',
         'counts': manifest.groupby(['split', 'label']).size().to_dict()}
audit['counts'] = {str(k): int(v) for k, v in audit['counts'].items()}
(OUT / 'data_audit.json').write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2), flush=True)

# Letterbox keeps the entire radiograph instead of cropping peripheral anatomy.
class Letterbox:
    def __call__(self, im):
        return ImageOps.pad(im, (224, 224), method=Image.Resampling.BICUBIC, color=0)

normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
eval_transform = transforms.Compose([Letterbox(), transforms.ToTensor(), normalize])
train_transform = transforms.Compose([Letterbox(),
    transforms.RandomAffine(degrees=7, translate=(.03, .03), scale=(.95, 1.05)),
    transforms.ToTensor(), normalize])

class Radiographs(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(row.path) as im:
            x = self.transform(im.convert('RGB'))
        return x, torch.tensor(row.label, dtype=torch.float32)

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def loader(frame, training=False):
    return DataLoader(Radiographs(frame, train_transform if training else eval_transform),
        batch_size=BATCH_SIZE, shuffle=training, num_workers=2, pin_memory=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))

train_loader, val_loader, test_loader = loader(train, True), loader(val), loader(test)
@torch.no_grad()
def predict(data_loader):
    model.eval()
    truth, probabilities = [], []
    for x, y in data_loader:
        with torch.autocast(device_type='cuda'):
            logits = model(x.to(DEVICE)).flatten()
        probabilities.extend(torch.sigmoid(logits.float()).cpu().numpy())
        truth.extend(y.numpy())
    return np.asarray(truth, dtype=int), np.asarray(probabilities)

# Three independent seeds; use development data only to choose ensemble or single model.
started = time.time()
member_paths, member_val, all_history = [], [], []
for member_index, member_seed in enumerate([42, 123, 2026]):
    if TRAIN_DEADLINE - time.monotonic() < 120:
        break
    random.seed(member_seed)
    np.random.seed(member_seed)
    torch.manual_seed(member_seed)
    torch.cuda.manual_seed_all(member_seed)
    train_loader = loader(train, True)
    member_deadline = time.monotonic() + (TRAIN_DEADLINE-time.monotonic())/(3-member_index)
    checkpoint = OUT / f'best_seed_{member_seed}.pt'
    print('Training ensemble member', member_seed, flush=True)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(DEVICE)
    ratio = float((train.label == 0).sum() / (train.label == 1).sum())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio], device=DEVICE))
    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 5e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-4}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)
    history, best_auc, stale = [], -1., 0
    
    for epoch in range(EPOCHS):
        if time.monotonic() >= member_deadline:
            print('Training time budget reached.', flush=True)
            break
        model.train()
        total_loss = 0.
        seen = 0
        for x, y in train_loader:
            if time.monotonic() >= member_deadline:
                break
            seen += len(y)
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                logits = model(x).flatten()
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
        vy, vp = predict(val_loader)
        auc = float(roc_auc_score(vy, vp))
        record = {'seed': member_seed, 'epoch': epoch+1, 'train_loss': total_loss/max(seen, 1),
                  'val_auroc': auc, 'val_average_precision': float(average_precision_score(vy, vp))}
        scheduler.step(auc)
        history.append(record)
        pd.DataFrame(history).to_csv(OUT / f'history_seed_{member_seed}.csv', index=False)
        print(record, flush=True)
        if auc > best_auc + 1e-5:
            best_auc, stale = auc, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= PATIENCE:
            print('Early stopping.', flush=True)
            break
    
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    vy, vp = predict(val_loader)
    member_paths.append(str(checkpoint))
    member_val.append(vp)
    all_history.extend(history)
    del optimizer, scaler, model
    torch.cuda.empty_cache()
if not member_paths:
    raise RuntimeError('No ensemble member completed within the training budget.')
candidates = member_val + [np.mean(member_val, axis=0)]
candidate_auc = [float(roc_auc_score(vy, p)) for p in candidates]
selected = int(np.argmax(candidate_auc))
selected_indices = list(range(len(member_paths))) if selected == len(member_paths) else [selected]
vp = candidates[selected]
best_auc = candidate_auc[selected]
history = all_history
pd.DataFrame(history).to_csv(OUT / 'history.csv', index=False)
selection = {'candidate_validation_auroc': candidate_auc, 'selected_member_indices': selected_indices,
             'selection_rule': 'maximum validation AUROC among single members and equal-weight ensemble',
             'checkpoints': [Path(p).name for p in member_paths],
             'research_status': 'exploratory follow-up; original test results already inspected'}
(OUT / 'ensemble_selection.json').write_text(json.dumps(selection, indent=2))
print('Validation selection:', selection, flush=True)
fpr, tpr, thresholds = roc_curve(vy, vp)
valid = np.isfinite(thresholds)
threshold = float(thresholds[valid][np.argmax((tpr-fpr)[valid])])
# Threshold is chosen only from validation (maximum Youden J), then frozen.
test_probabilities = []
for i in selected_indices:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(DEVICE)
    model.load_state_dict(torch.load(member_paths[i], map_location=DEVICE, weights_only=True))
    ty, member_tp = predict(test_loader)
    test_probabilities.append(member_tp)
    del model
    torch.cuda.empty_cache()
tp = np.mean(test_probabilities, axis=0)

def scores(y, p, cutoff):
    pred = (p >= cutoff).astype(int)
    tn, fp, fn, tpos = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'auroc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'f1': float(f1_score(y, pred, zero_division=0)),
            'sensitivity': float(tpos/(tpos+fn)), 'specificity': float(tn/(tn+fp)),
            'precision': float(tpos/(tpos+fp)) if tpos+fp else 0.0}

# Cluster bootstrap resamples filename-derived subjects, retaining all their images.
rng = np.random.default_rng(SEED)
test = test.reset_index(drop=True)
groups = [np.flatnonzero(test.patient.to_numpy() == g) for g in test.patient.unique()]
boot = []
for _ in range(1000):
    idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    if len(np.unique(ty[idx])) == 2:
        boot.append(scores(ty[idx], tp[idx], threshold))
ci = {key: np.percentile([row[key] for row in boot], [2.5, 97.5]).tolist() for key in boot[0]}
result = {'seed': SEED, 'threshold': threshold, 'threshold_rule': 'validation Youden J',
          'test': scores(ty, tp, threshold), 'test_at_0.5': scores(ty, tp, .5),
          'test_cluster_bootstrap_95ci': ci, 'bootstrap_replicates': len(boot),
          'validation': scores(vy, vp, threshold),
          'training_seconds': time.time()-started, 'best_validation_auroc': best_auc,
          'test_majority_class_accuracy': float(max(np.mean(ty), 1-np.mean(ty)))}
(OUT / 'metrics.json').write_text(json.dumps(result, indent=2))
for name, frame, y, p in [('validation', val, vy, vp), ('test', test, ty, tp)]:
    frame.assign(y_true=y, probability=p, prediction=(p >= threshold).astype(int)).to_csv(
        OUT / (name + '_predictions.csv'), index=False)
(OUT / 'inference_config.json').write_text(json.dumps({
    'architecture': 'efficientnet_b0', 'threshold': threshold,
    'selected_checkpoints': [Path(member_paths[i]).name for i in selected_indices],
    'aggregation': 'arithmetic mean of sigmoid probabilities',
    'classes': ['NORMAL', 'PNEUMONIA'],
    'preprocessing': 'RGB; bicubic letterbox 224; ToTensor; ImageNet mean/std'}, indent=2))
environment = {'python': sys.version, 'torch': torch.__version__,
    'torchvision': torchvision.__version__, 'numpy': np.__version__,
    'pandas': pd.__version__, 'sklearn': sklearn.__version__,
    'gpu': torch.cuda.get_device_name(0), 'epochs_requested': EPOCHS,
    'batch_size': BATCH_SIZE, 'seed': SEED, 'training_budget_minutes': 110, 'seeds': [42, 123, 2026]}
(OUT / 'environment.json').write_text(json.dumps(environment, indent=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
ConfusionMatrixDisplay.from_predictions(ty, tp >= threshold,
    display_labels=['Normal', 'Pneumonia'], ax=axes[0], colorbar=False)
fpr, tpr, _ = roc_curve(ty, tp)
axes[1].plot(fpr, tpr, label=f"AUROC={result['test']['auroc']:.3f}")
axes[1].plot([0, 1], [0, 1], '--', color='gray')
axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Test ROC')
axes[1].legend()
precision, recall, _ = precision_recall_curve(ty, tp)
axes[2].plot(recall, precision, label=f"AP={result['test']['average_precision']:.3f}")
axes[2].axhline(np.mean(ty), linestyle='--', color='gray')
axes[2].set(xlabel='Recall', ylabel='Precision', title='Test precision–recall')
axes[2].legend()
fig.tight_layout()
fig.savefig(OUT / 'test_evaluation.png', dpi=180)
plt.show()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
hist = pd.DataFrame(history)
for seed, h in hist.groupby('seed'):
    axes[0].plot(h.epoch, h.train_loss, label=str(seed))
axes[0].legend()
axes[0].set(xlabel='Epoch', ylabel='Weighted loss')
for seed, h in hist.groupby('seed'):
    axes[1].plot(h.epoch, h.val_auroc, label=str(seed))
axes[1].legend()
axes[1].set(xlabel='Epoch', ylabel='Validation AUROC')
fig.tight_layout()
fig.savefig(OUT / 'learning_curves.png', dpi=180)
plt.show()
print(json.dumps(result, indent=2), flush=True)
print('DONE. Outputs saved to', OUT, flush=True)
