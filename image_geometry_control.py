"""Post hoc image-dimension control; fits training data only with fixed settings."""
import hashlib
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPECTED={'train':(3815,'293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d'),
          'test':(618,'c0de6a4a18c8c295d85551b5b8602410e2e26d173b8ab9f141e82cd6fdca9378')}


def features(frame):
    values=frame[['width','height']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Invalid image dimensions')
    return np.log(values)


def fit_control(frame):
    train=frame.loc[frame.split=='train']
    if set(train.label)!={0,1}:
        raise ValueError('Both training classes required')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,
        class_weight=None,random_state=42))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(features(train),train.label.to_numpy())
    scaler=model.named_steps['standardscaler'];classifier=model.named_steps['logisticregression']
    config=dict(features=['natural_log_width','natural_log_height'],fit_partition='train',
        standardization_mean=scaler.mean_.tolist(),standardization_scale=scaler.scale_.tolist(),
        coefficient=classifier.coef_[0].tolist(),intercept=float(classifier.intercept_[0]),
        classes=classifier.classes_.tolist(),C=1.0,solver='lbfgs',max_iter=1000,
        class_weight=None,threshold=0.5,temperature_fitting=False,hyperparameter_search=False,
        iterations=int(classifier.n_iter_[0]))
    return model,config


def evaluate(frame,probability):
    y=frame.label.to_numpy();pred=probability>=.5
    tn=int(np.sum((y==0)&~pred));fp=int(np.sum((y==0)&pred))
    fn=int(np.sum((y==1)&~pred));tp=int(np.sum((y==1)&pred))
    return dict(images=len(frame),filename_proxy_groups=int(frame.patient.nunique()),
        auroc=float(roc_auc_score(y,probability)),average_precision=float(average_precision_score(y,probability)),
        accuracy=float(np.mean(y==pred)),sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),
        brier_score=float(np.mean((y-probability)**2)),tn=tn,fp=fp,fn=fn,tp=tp)


def run(model_dir,data_root,out_dir):
    model_dir=Path(model_dir);data_root=Path(data_root);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(model_dir/'split_manifest.csv')
    required={'sha256','patient','filename','original_split','label','split'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any() or frame.sha256.duplicated().any():
        raise ValueError('Invalid frozen manifest')
    if frame.groupby('patient').split.nunique().max()!=1:
        raise ValueError('Proxy-group overlap')
    if frame.split.value_counts().to_dict()!={'train':3815,'selection':456,'calibration':482,'test':618}:
        raise ValueError('Unexpected partition counts')
    for split,(count,digest) in EXPECTED.items():
        subset=frame.loc[frame.split==split]
        if len(subset)!=count or hashlib.sha256('\n'.join(sorted(subset.sha256)).encode()).hexdigest()!=digest:
            raise ValueError('Frozen cohort digest mismatch: '+split)
    widths=[];heights=[]
    for i,row in enumerate(frame.itertuples()):
        path=data_root/row.original_split/('NORMAL' if row.label==0 else 'PNEUMONIA')/row.filename
        with Image.open(path) as image:
            gray=image.convert('L')
            digest=hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()
            if digest!=row.sha256:
                raise ValueError('Pixel identity mismatch: '+row.filename)
            width,height=image.size
        widths.append(width);heights.append(height)
        if (i+1)%1000==0:
            print('Geometry identity checks:',i+1,flush=True)
    frame['width']=widths;frame['height']=heights
    model,config=fit_control(frame)
    # Persist the fixed training-only model before any nontraining performance is evaluated.
    (out/'geometry_model.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    report=dict(status='Post hoc descriptive dimension-only control; not a clinical classifier',
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
        configuration=config,decoded_pixels_used_only_for_identity_check=True,
        radiographic_pixels_used_as_predictor_features=False,original_models_or_thresholds_changed=False,
        metrics={},geometry_summary={},
        limitations=['Specified after the neural-network test results were inspected; exploratory, not confirmatory.',
            'Dimension predictability can reflect acquisition, processing or cohort selection; it does not identify the cause.',
            'This control cannot establish whether the neural networks actually use dimensions or related shortcuts.',
            'Width and height are not diagnostic findings. Filename groups are unverified patient proxies.',
            'No feature or hyperparameter search, threshold tuning, calibration fit or external evaluation.'])
    frames=[]
    for split in ('train','selection','calibration','test'):
        part=frame.loc[frame.split==split].copy();p=model.predict_proba(features(part))[:,1]
        report['metrics'][split]=evaluate(part,p)
        report['geometry_summary'][split]={str(label):{
            'images':len(group),'width_percentiles_25_50_75':np.percentile(group.width,[25,50,75]).tolist(),
            'height_percentiles_25_50_75':np.percentile(group.height,[25,50,75]).tolist()}
            for label,group in part.groupby('label')}
        part['geometry_score']=p;part['geometry_prediction']=(p>=.5).astype(int);frames.append(part)
    pd.concat(frames).to_csv(out/'geometry_predictions.csv',index=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('GEOMETRY_CONTROL',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    root=Path('/kaggle/input')
    models=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_benchmark_outputs']
    roots=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
    if len(models)!=1 or not roots:
        raise ValueError('Attach one completed 224 benchmark output and the original image dataset')
    run(models[0],roots[0],'/kaggle/working/geometry_control_outputs')
