"""Verify the secondary experiment's group separation before submitting it."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research_benchmark import split_cohorts
from selection import internal_split

source = pd.read_csv(ROOT / 'artifacts/baseline_v1/pneumonia_outputs/split_manifest.csv')
frames = split_cohorts(source.drop(columns='split'))
frame = frames['train'].sort_values('sha256').reset_index(drop=True)
folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=20261115).split(np.zeros(len(frame)), frame.label, frame.patient))
seen = set(); records = []
for fold, (train, test) in enumerate(folds):
    fit, stop = internal_split(frame.iloc[train].reset_index(drop=True))
    held = frame.iloc[test]
    assert not set(fit.patient) & set(stop.patient)
    assert not (set(fit.patient) | set(stop.patient)) & set(held.patient)
    assert not seen & set(held.sha256)
    seen.update(held.sha256)
    assert set(fit.sha256) | set(stop.sha256) | set(held.sha256) == set(frame.sha256)
    for other in ['selection', 'calibration', 'test']:
        assert not set(frame.patient) & set(frames[other].patient)
    records.append(dict(fold=fold, fit_images=len(fit), internal_images=len(stop), outer_images=len(held),
                        outer_groups=int(held.patient.nunique()), outer_normal=int(np.sum(held.label == 0)),
                        outer_pneumonia=int(np.sum(held.label == 1))))
assert seen == set(frame.sha256)
for file in ['run_grouped_cv.py', 'build_grouped_cv.py', 'submitted_grouped_cv.py']:
    compile((HERE / file).read_text(encoding='utf-8'), file, 'exec')
report = dict(status='passed', scope='All outer/inner group partitions and complete OOF coverage, source syntax; no Kaggle run or new performance result',
              images=len(frame), groups=int(frame.patient.nunique()), folds=records,
              source_sha256_lf=hashlib.sha256((HERE / 'submitted_grouped_cv.py').read_text(encoding='utf-8').encode()).hexdigest())
(HERE / 'cv_local_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
