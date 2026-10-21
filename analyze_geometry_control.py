"""Verify the dimension-control export and estimate frozen paired uncertainty."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from analyze_benchmark_comparison import analyze
from compare_experiments import metrics


def reconstruct(directory):
    directory=Path(directory)
    config=json.loads((directory/'geometry_model.json').read_text(encoding='utf-8'))
    frame=pd.read_csv(directory/'geometry_predictions.csv',float_precision='round_trip')
    required={'sha256','patient','label','split','width','height','geometry_score','geometry_prediction'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any() or frame.sha256.duplicated().any():
        raise ValueError('Invalid geometry export identities')
    if config['features']!=['natural_log_width','natural_log_height'] or config['classes']!=[0,1] or config['threshold']!=.5:
        raise ValueError('Geometry configuration changed')
    values=frame[['width','height']].to_numpy(float)
    mean=np.asarray(config['standardization_mean'],dtype=float)
    scale=np.asarray(config['standardization_scale'],dtype=float)
    coefficient=np.asarray(config['coefficient'],dtype=float)
    if not np.isfinite(values).all() or np.any(values<=0) or mean.shape!=(2,) or scale.shape!=(2,) or coefficient.shape!=(2,):
        raise ValueError('Invalid geometry feature configuration')
    if not np.isfinite(np.r_[mean,scale,coefficient,config['intercept']]).all() or np.any(scale<=0):
        raise ValueError('Invalid geometry standardization or coefficients')
    score=expit(((np.log(values)-mean)/scale)@coefficient+config['intercept'])
    if not np.isfinite(frame.geometry_score).all() or not frame.geometry_score.between(0,1).all():
        raise ValueError('Invalid saved geometry scores')
    difference=np.abs(score-frame.geometry_score.to_numpy())
    changed=int(np.sum((score>=.5).astype(int)!=frame.geometry_prediction.to_numpy()))
    if difference.max()>1e-12 or changed:
        raise ValueError('Geometry export reconstruction failed')
    frame=frame.rename(columns={'label':'y_true','geometry_score':'probability','geometry_prediction':'prediction'})
    return frame,dict(status='passed',images=len(frame),max_absolute_score_difference=float(difference.max()),
        changed_classifications=changed,score_tolerance=1e-12,
        scope='Reconstruction from saved dimensions and coefficients; original pixel identities were checked in the control run')


def run(control,neural,out):
    control=Path(control);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frame,verification=reconstruct(control)
    if frame.split.value_counts().to_dict()!={'train':3815,'selection':456,'calibration':482,'test':618}:
        raise ValueError('Unexpected geometry cohort counts')
    summary=json.loads((control/'summary.json').read_text(encoding='utf-8'))
    for split,part in frame.groupby('split'):
        for key,value in metrics(part).items():
            if not np.isclose(value,summary['metrics'][split][key],rtol=1e-10,atol=1e-12):
                raise ValueError('Geometry metric reconstruction mismatch: '+split+' '+key)
    export=out/'verified_geometry';export.mkdir(exist_ok=True)
    frame.loc[frame.split=='test'].to_csv(export/'test_predictions.csv',index=False)
    comparison=analyze(neural,export,out/'paired',replicates=1000)
    result=dict(status='Completed frozen geometry-control analysis; exploratory',
        export_verification=verification,comparison=comparison,
        mapping={'baseline':'Frozen selected 320-pixel EfficientNet-B0 ensemble','followup':'Fixed image-dimension logistic control','difference':'geometry minus neural; lower Brier is better'},
        limitations=['Intervals are conditional on fitted models and filename-proxy grouping; no training or selection uncertainty.',
                     'Control was specified after neural-network test results; it is not confirmatory.',
                     'Dimension predictability does not establish that the neural networks rely on dimensions.',
                     'No external evaluation, threshold adjustment or model selection.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('GEOMETRY_ANALYSIS',json.dumps(result),flush=True)
    return result
