"""Reproduce discrimination, calibration and error summaries from frozen predictions.

This is descriptive evaluation; it never fits a threshold or changes model weights.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def analyze(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    candidates = [('validation', '#2374AB'), ('selection', '#2374AB'),
                  ('calibration', '#498339'), ('test', '#D45D31')]
    available = [(split, color) for split, color in candidates
                 if (directory/f'{split}_predictions.csv').is_file()]
    if 'test' not in [split for split, _ in available] or len(available) < 2:
        raise ValueError('Expected test predictions and at least one development partition')
    results['interpretation'] = ('Descriptive frozen-score diagnostics. Selection and calibration '
        'partitions influenced model fitting or selection; their curves are not unbiased evaluation.')
    for split, color in available:
        df = pd.read_csv(directory/f'{split}_predictions.csv')
        y, p = df.y_true.to_numpy(), df.probability.to_numpy()
        if set(y) != {0, 1} or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Invalid labels or probabilities in ' + split)
        results[split] = {'n': len(df), 'auroc': float(roc_auc_score(y,p)),
            'average_precision': float(average_precision_score(y,p)),
            'brier_score': float(brier_score_loss(y,p)),
            'log_loss': float(log_loss(y,np.clip(p,1e-7,1-1e-7)))}
        for label, axis in [(0,axes[0]),(1,axes[1])]:
            axis.hist(p[y==label], bins=np.linspace(0,1,21), weights=np.ones(sum(y==label))/sum(y==label),
                      histtype='step', linewidth=2, color=color, label=split)
        edges = np.linspace(0,1,11)
        bucket = np.minimum(np.digitize(p,edges)-1,9)
        points = [(float(p[bucket==i].mean()),float(y[bucket==i].mean()),int(sum(bucket==i)))
                  for i in range(10) if np.any(bucket==i)]
        axes[2].plot([x[0] for x in points],[x[1] for x in points], 'o-',color=color,label=split)
        results[split]['calibration_bins'] = points
        results[split]['ece_10_equal_width_bins'] = float(sum(n*abs(a-b) for a,b,n in points)/len(y))
        for label in [0,1]:
            rows=df[df.y_true==label]
            results[split][f'class_{label}_mean_score']=float(rows.probability.mean())
        errors=df[df.y_true!=df.prediction].copy()
        errors.to_csv(output/f'{split}_errors.csv',index=False)
    axes[0].set(title='Normal images',xlabel='Pneumonia score',ylabel='Fraction of class')
    axes[1].set(title='Pneumonia images',xlabel='Pneumonia score',ylabel='Fraction of class')
    axes[2].plot([0,1],[0,1],'--',color='gray')
    axes[2].set(title='Reliability diagram (descriptive)',xlabel='Mean score in bin',ylabel='Observed positive fraction',xlim=(0,1),ylim=(0,1))
    for axis in axes: axis.legend()
    fig.tight_layout()
    fig.savefig(output/'score_shift_and_calibration.png',dpi=180)
    plt.close(fig)
    (output/'prediction_analysis.json').write_text(json.dumps(results,indent=2))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.directory,args.output),indent=2))
