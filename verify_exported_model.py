"""Independently reload exported checkpoints and reproduce the entire test cohort on CPU."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torchvision import models, transforms

from predict import load_config
from validate_benchmark import validate_benchmark, temperature_scale


def model_for(architecture):
    if architecture=='efficientnet_b0':
        model=models.efficientnet_b0(weights=None)
        model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,1)
    elif architecture=='densenet121':
        model=models.densenet121(weights=None)
        model.classifier=torch.nn.Linear(model.classifier.in_features,1)
    elif architecture=='resnet18':
        model=models.resnet18(weights=None)
        model.fc=torch.nn.Linear(model.fc.in_features,1)
    else: raise ValueError('Unsupported architecture')
    return model


def verify(model_dir,data_root,output):
    model_dir,data_root,output=map(Path,[model_dir,data_root,output])
    output.mkdir(parents=True,exist_ok=True)
    consistency=validate_benchmark(model_dir)
    print('Saved artifact consistency passed',flush=True)
    config,paths=load_config(model_dir); size=config.get('image_size',224)
    frame=pd.read_csv(model_dir/'test_predictions.csv').sort_values('sha256').reset_index(drop=True)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    # Cache only the 618 test tensors, keeping source identity checks independent of prediction values.
    images=[]
    for row in frame.itertuples():
        path=data_root/row.original_split/('NORMAL' if row.y_true==0 else 'PNEUMONIA')/row.filename
        with Image.open(path) as image:
            gray=image.convert('L')
            digest=hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()
            if digest!=row.sha256: raise ValueError('Source image hash mismatch: '+row.filename)
            resized=ImageOps.pad(image.convert('RGB'),(size,size),method=Image.Resampling.BICUBIC,color=0)
            images.append(transform(resized))
    print('Verified decoded pixel identity for',len(images),'test images',flush=True)
    torch.set_num_threads(2)
    checkpoint_records=[]
    results=json.loads((model_dir/'all_trial_results.json').read_text())
    for record in results:
        path=model_dir/record['trial']/'best_weights.pt'
        model=model_for(record['architecture'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.eval()
        with torch.inference_mode(): prediction=model(torch.stack(images[:2]))
        if prediction.shape!=(2,1) or not torch.isfinite(prediction).all():
            raise ValueError('Checkpoint forward failure: '+record['trial'])
        checkpoint_records.append(dict(trial=record['trial'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                       strict_load_and_forward=True))
        print('Checkpoint passed:',record['trial'],flush=True)
        del model
    member_scores=[]
    for path in paths:
        model=model_for(config['architecture'])
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.eval(); predictions=[]
        with torch.inference_mode():
            for start in range(0,len(images),16):
                batch=torch.stack(images[start:start+16])
                predictions.extend(torch.sigmoid(model(batch).flatten()).numpy())
        member_scores.append(np.asarray(predictions,dtype=float)); del model
        print('Completed CPU test inference:',path.parent.name,flush=True)
    raw=np.mean(member_scores,axis=0); calibrated=temperature_scale(raw,config.get('temperature',1.))
    difference=np.abs(calibrated-frame.probability.to_numpy())
    predictions=(calibrated>=config['threshold']).astype(int)
    changed=int(np.sum(predictions!=frame.prediction.to_numpy()))
    passed=bool(np.isfinite(calibrated).all() and difference.max()<=1e-4 and changed==0)
    frame.assign(cpu_probability=calibrated,cpu_prediction=predictions,absolute_score_difference=difference).to_csv(output/'cpu_reproduction.csv',index=False)
    report=dict(status='passed' if passed else 'numerical discrepancy requires review',
        images=len(frame),image_size=size,architecture=config['architecture'],
        max_absolute_score_difference=float(difference.max()),mean_absolute_score_difference=float(difference.mean()),
        changed_predictions=changed,score_tolerance=1e-4,artifact_consistency=consistency,
        checkpoints=checkpoint_records,torch_version=torch.__version__,
        interpretation='Engineering reproduction on the same test cohort, not independent external or clinical validation')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    print('DONE',json.dumps(report),flush=True)
    if not passed: raise RuntimeError('CPU numerical reproduction requires review; see saved report')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); verify(args.model_dir,args.data_root,args.output)
