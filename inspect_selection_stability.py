"""Descriptive development-group bootstrap of fixed architecture/resolution scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from select_resolution import select


def inspect(low,high,replicates=1000):
    if replicates<1: raise ValueError('replicates must be positive')
    decision=select(low,high)  # Validates both development cohorts and every member prediction.
    candidates=sorted(decision['candidates'],key=lambda c:(c['image_size'],c['architecture']))
    names=[f"{c['image_size']}:{c['architecture']}" for c in candidates]
    arrays=[];reference=None
    for candidate in candidates:
        root=Path(low if candidate['image_size']==224 else high)
        members=[]
        for seed in [42,123,2026]:
            frame=pd.read_csv(root/f"{candidate['architecture']}_seed_{seed}"/'selection_predictions.csv').sort_values('sha256').reset_index(drop=True)
            if reference is None: reference=frame
            members.append(frame.probability.to_numpy(float))
        arrays.append(members)
    scores=np.asarray(arrays);y=reference.y_true.to_numpy(int)
    groups=[np.flatnonzero(reference.patient.to_numpy()==g) for g in reference.patient.unique()]
    rng=np.random.default_rng(42);counts=Counter();sample_scores=[];ties=0
    for _ in range(replicates):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        labels=y[indices]
        if len(np.unique(labels))!=2: continue
        means=np.array([np.mean([roc_auc_score(labels,member[indices]) for member in members]) for members in scores])
        winner=int(np.argmax(means))  # Candidate order implements the frozen tie rule.
        counts[names[winner]]+=1
        ties+=int(np.count_nonzero(means==means.max())>1)
        sample_scores.append(means)
    if not sample_scores: raise ValueError('No usable two-class bootstrap samples')
    sample_scores=np.asarray(sample_scores);used=len(sample_scores)
    original=decision['selected'];chosen=f"{original['image_size']}:{original['architecture']}"
    return dict(status='Exploratory development-sample sensitivity analysis',
        images=len(y),filename_proxy_groups=len(groups),requested_replicates=replicates,
        usable_replicates=used,replicates_with_tied_maximum=ties,
        original_development_choice=chosen,
        winner_counts={name:counts[name] for name in names},
        winner_fractions={name:counts[name]/used for name in names},
        mean_selection_auroc_percentile_95={name:np.percentile(sample_scores[:,i],[2.5,97.5]).tolist() for i,name in enumerate(names)},
        test_predictions_accessed=False,changes_to_deployed_choice=False,
        interpretation='Resamples the same selection groups jointly for all fixed models. Winner frequencies are descriptive stability measures, not posterior probabilities, superiority probabilities or external performance estimates. Checkpoints were selected using this partition, so selection optimism remains. Training and checkpoint selection are not repeated; three fixed seeds are averaged without seed resampling. The frozen original choice is retained.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('low',type=Path);parser.add_argument('high',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--replicates',type=int,default=1000)
    args=parser.parse_args();report=inspect(args.low,args.high,args.replicates)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
