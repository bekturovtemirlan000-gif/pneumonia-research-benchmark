"""Post hoc error description from frozen predictions; never selects or fits models."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def subgroup(path, label):
    name=Path(str(path).replace('\\','/')).name.lower()
    if label==0:
        return 'normal_source_label'
    if '_bacteria_' in name:
        return 'pneumonia_bacteria_filename'
    if '_virus_' in name:
        return 'pneumonia_virus_filename'
    return 'pneumonia_unspecified_filename'


def load_predictions(directory):
    directory=Path(directory)
    frame=pd.read_csv(directory/'test_predictions.csv',float_precision='round_trip')
    required={'sha256','patient','path','y_true','probability','prediction'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('Missing prediction identity or score fields')
    if frame.sha256.duplicated().any() or set(frame.y_true)!={0,1}:
        raise ValueError('Expected unique images and both binary source classes')
    if not set(frame.prediction).issubset({0,1}) or not np.isfinite(frame.probability).all() or not frame.probability.between(0,1).all():
        raise ValueError('Invalid frozen predictions')
    frame=frame.sort_values('sha256').reset_index(drop=True)
    frame['filename']=frame.path.map(lambda p:Path(str(p).replace('\\','/')).name)
    frame['source_subgroup']=[subgroup(p,y) for p,y in zip(frame.path,frame.y_true)]
    return frame


def counts(frame):
    y=frame.y_true.to_numpy();pred=frame.prediction.to_numpy()
    tn=int(np.sum((y==0)&(pred==0)));fp=int(np.sum((y==0)&(pred==1)))
    fn=int(np.sum((y==1)&(pred==0)));tp=int(np.sum((y==1)&(pred==1)))
    return dict(images=len(frame),filename_proxy_groups=int(frame.patient.nunique()),
        tn=tn,fp=fp,fn=fn,tp=tp,
        sensitivity=tp/(tp+fn) if tp+fn else None,
        specificity=tn/(tn+fp) if tn+fp else None,
        brier_score=float(np.mean((y-frame.probability.to_numpy())**2)))


def inspect(low_dir,high_dir,out_dir):
    low=load_predictions(low_dir);high=load_predictions(high_dir)
    for column in ('sha256','patient','filename','y_true','source_subgroup'):
        if not low[column].equals(high[column]):
            raise ValueError('Frozen cohort mismatch: '+column)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    a=low.prediction.to_numpy()==low.y_true.to_numpy()
    b=high.prediction.to_numpy()==high.y_true.to_numpy()
    detail=low[['sha256','patient','filename','y_true','source_subgroup']].copy()
    detail['prediction_224']=low.prediction;detail['prediction_320']=high.prediction
    detail['score_224']=low.probability;detail['score_320']=high.probability
    detail['correct_224']=a;detail['correct_320']=b
    detail.loc[~a|~b].to_csv(out/'frozen_error_cases.csv',index=False)
    report=dict(status='Post hoc descriptive error analysis; no model or cutoff changes',
        images=len(low),filename_proxy_groups=int(low.patient.nunique()),
        overlap=dict(both_correct=int(np.sum(a&b)),both_incorrect=int(np.sum(~a&~b)),
                     only_224_correct=int(np.sum(a&~b)),only_320_correct=int(np.sum(~a&b)),
                     different_binary_predictions=int(np.sum(low.prediction!=high.prediction))),
        models={},subgroups={},
        limitations=['Subgroups are inferred from source filenames, not independently confirmed etiology.',
                     'The model is binary pneumonia-label classification, not bacterial-versus-viral diagnosis.',
                     'Post hoc source-test description; no confirmatory subgroup comparisons or external validity.',
                     'Filename groups are proxies; image-level rates do not establish independent patient-level performance.',
                     'No age, sex or hospital subgroup information is inferred. Small error counts do not guarantee future sensitivity.'])
    for size,frame in ((224,low),(320,high)):
        report['models'][str(size)]=counts(frame)
        report['subgroups'][str(size)]={name:counts(part) for name,part in frame.groupby('source_subgroup')}
    report['overlap_by_source_label']={}
    for label in (0,1):
        mask=low.y_true.to_numpy()==label
        report['overlap_by_source_label'][str(label)]=dict(images=int(mask.sum()),
            both_correct=int(np.sum(a&b&mask)),both_incorrect=int(np.sum(~a&~b&mask)),
            only_224_correct=int(np.sum(a&~b&mask)),only_320_correct=int(np.sum(~a&b&mask)))
    memberships=low.groupby('patient').source_subgroup.nunique()
    report['proxy_groups_spanning_filename_subgroups']=int((memberships>1).sum())
    report['limitations'].append('Subgroup proxy-group counts can overlap and must not be summed as independent patients.')
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    root=Path('/kaggle/input')
    def locate(name):
        matches=[p.parent for p in root.rglob('inference_config.json') if p.parent.name==name]
        if len(matches)!=1:
            raise ValueError('Attach exactly one completed output: '+name)
        return matches[0]
    report=inspect(locate('pneumonia_benchmark_outputs'),locate('pneumonia_resolution320_outputs'),
                   '/kaggle/working/frozen_error_analysis_outputs')
    if report['images']!=618 or report['filename_proxy_groups']!=426:
        raise ValueError('Unexpected frozen test cohort')
    print('FROZEN_ERROR_ANALYSIS',json.dumps(report),flush=True)
