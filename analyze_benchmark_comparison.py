"""Descriptive paired uncertainty for already-frozen source test predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_experiments import compare, metrics


def analyze(previous, benchmark, output, replicates=1000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    comparison=compare(previous,benchmark,replicates)
    frame=pd.read_csv(Path(benchmark)/'test_predictions.csv').sort_values('sha256').reset_index(drop=True)
    groups=[np.flatnonzero(frame.patient.to_numpy()==g) for g in frame.patient.unique()]
    rng=np.random.default_rng(42); samples=[]
    for _ in range(replicates):
        indices=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))])
        sample=frame.iloc[indices]
        if sample.y_true.nunique()==2: samples.append(metrics(sample))
    if not samples: raise ValueError('No usable bootstrap samples')
    intervals={name:np.percentile([m[name] for m in samples],[2.5,97.5]).tolist() for name in samples[0]}
    report={'status':'Exploratory; the original source test was previously inspected',
            'benchmark_metrics':metrics(frame),'benchmark_cluster_bootstrap_95ci':intervals,
            'bootstrap_replicates':len(samples),'paired_comparison':comparison,
            'interpretation':'Conditional test-sampling uncertainty only. Does not capture training, model selection, proxy-group error or external generalization. No threshold refitting or model selection occurs.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2))
    print('COMPARISON_DONE',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous',type=Path);parser.add_argument('benchmark',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.previous,args.benchmark,args.output)
