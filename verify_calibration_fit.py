"""Recompute calibration-only scores on CPU; never replace the frozen model config."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_curve
from torchvision import transforms

from predict import load_config
from verify_exported_model import model_for


def fit_calibration(y, raw):
    z=logit(np.clip(np.asarray(raw,dtype=float),1e-7,1-1e-7))
    y=np.asarray(y,dtype=int)
    def objective(log_temperature):
        scaled=z/np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0,scaled)-y*scaled))
    fit=minimize_scalar(objective,bounds=(-3,3),method='bounded')
    if not fit.success: raise ValueError('Temperature optimization failed')
    temperature=float(np.exp(fit.x)); scores=expit(z/temperature)
    fpr,tpr,thresholds=roc_curve(y,scores)
    finite=np.isfinite(thresholds)
    threshold=float(thresholds[finite][np.argmax((tpr-fpr)[finite])])
    return temperature,threshold,scores


def calibration_frame(model_dir):
    model_dir=Path(model_dir)
    frame=pd.read_csv(model_dir/'calibration_predictions.csv').sort_values('sha256').reset_index(drop=True)
    manifest=pd.read_csv(model_dir/'split_manifest.csv')
    expected=manifest.loc[manifest['split']=='calibration']
    expected_ids=set(expected[['path','label','patient','sha256']].itertuples(index=False,name=None))
    observed_ids=set(frame[['path','y_true','patient','sha256']].itertuples(index=False,name=None))
    if len(frame)!=482 or len(expected)!=482 or frame.sha256.duplicated().any() or expected_ids!=observed_ids:
        raise ValueError('Expected the frozen 482-image calibration partition')
    return frame


def verify(model_dir,data_root,output):
    model_dir,data_root,output=map(Path,(model_dir,data_root,output))
    output.mkdir(parents=True,exist_ok=True)
    config,checkpoints=load_config(model_dir)
    size=config.get('image_size',224)
    frame=calibration_frame(model_dir)
    torch.set_num_threads(2)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    images=[]
    for row in frame.itertuples():
        path=data_root/row.original_split/('NORMAL' if row.y_true==0 else 'PNEUMONIA')/row.filename
        with Image.open(path) as image:
            gray=image.convert('L')
            if hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()!=row.sha256:
                raise ValueError('Calibration pixel identity mismatch')
            images.append(transform(ImageOps.pad(image.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=0)))
    members=[];hashes=[]
    for checkpoint in checkpoints:
        model=model_for(config['architecture'])
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval();values=[]
        with torch.inference_mode():
            for start in range(0,len(images),16):
                values.extend(torch.sigmoid(model(torch.stack(images[start:start+16])).flatten()).numpy())
        members.append(np.asarray(values,dtype=float))
        hashes.append(dict(checkpoint=checkpoint.relative_to(model_dir).as_posix(),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
        print('CALIBRATION_MEMBER_DONE',size,checkpoint.parent.name,flush=True)
        del model
    raw=np.mean(members,axis=0)
    frozen_scores=expit(logit(np.clip(raw,1e-7,1-1e-7))/config['temperature'])
    temperature,threshold,refit_scores=fit_calibration(frame.y_true,raw)
    error=float(np.max(np.abs(frozen_scores-frame.probability.to_numpy())))
    changed=int(np.sum((frozen_scores>=config['threshold']).astype(int)!=frame.prediction.to_numpy()))
    relative_temperature_error=abs(temperature/config['temperature']-1)
    threshold_error=abs(threshold-config['threshold'])
    # Set before execution; numerical agreement, not a new model-selection criterion.
    passed=error<=1e-4 and changed==0 and relative_temperature_error<=1e-3 and threshold_error<=1e-4
    export=frame.copy()
    for checkpoint,values in zip(checkpoints,members): export['cpu_raw_'+checkpoint.parent.name]=values
    export['cpu_raw_mean']=raw;export['cpu_frozen_temperature_score']=frozen_scores
    export['cpu_refitted_temperature_score']=refit_scores
    export.to_csv(output/'calibration_cpu_scores.csv',index=False)
    report=dict(status='passed' if passed else 'numerical discrepancy requires review',image_size=size,
        architecture=config['architecture'],images=len(frame),checkpoints=hashes,
        saved_temperature=config['temperature'],cpu_refitted_temperature=temperature,
        relative_temperature_error=relative_temperature_error,saved_threshold=config['threshold'],
        cpu_refitted_threshold=threshold,absolute_threshold_error=threshold_error,
        max_frozen_score_difference=error,changed_frozen_calibration_predictions=changed,
        tolerances=dict(score_absolute=1e-4,temperature_relative=1e-3,threshold_absolute=1e-4),
        test_predictions_accessed=False,frozen_config_changed=False,torch_version=torch.__version__,
        interpretation='Calibration-only numerical reproduction from original pixels and saved checkpoints; no clinical or external validation. Fresh CPU scores are additional evidence, not original GPU calibration scores.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('CALIBRATION_VERIFICATION',json.dumps(report),flush=True)
    return report
