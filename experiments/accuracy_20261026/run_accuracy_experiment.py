"""Frozen-model accuracy experiment. Never choose a model or cutoff from test outcomes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from accuracy_selection import choose, accuracy_threshold, candidates
from predict_accuracy import model_for, image_tensor
from compare_experiments import metrics
from analyze_benchmark_comparison import analyze


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def aligned_manifest(low, high):
    fields=['sha256','patient','label','split','original_split','filename']
    frames=[pd.read_csv(p/'split_manifest.csv')[fields].sort_values('sha256').reset_index(drop=True) for p in [low,high]]
    if not frames[0].equals(frames[1]): raise ValueError('Resolution manifests disagree')
    frame=frames[0]
    if frame.isna().any().any() or frame.sha256.duplicated().any(): raise ValueError('Invalid manifest identities')
    expected={'train':3815,'selection':456,'calibration':482,'test':618}
    if frame.groupby('split').size().to_dict()!=expected: raise ValueError('Unexpected cohort sizes')
    if frame.groupby('patient').split.nunique().max()!=1: raise ValueError('Cross-partition group overlap')
    digest=hashlib.sha256('\n'.join(sorted(frame[frame.split=='test'].sha256)).encode()).hexdigest()
    if digest!='c0de6a4a18c8c295d85551b5b8602410e2e26d173b8ab9f141e82cd6fdca9378':
        raise ValueError('Frozen test cohort changed')
    return frame


def member_info(name, roots):
    size,trial=name.split('/')
    architecture,seed=trial.rsplit('_seed_',1)
    return dict(name=name,image_size=int(size),architecture=architecture,seed=int(seed),
                original_path=roots[int(size)]/trial/'best_weights.pt')


def selection_scores(frame, roots):
    expected=frame[frame.split=='selection'].reset_index(drop=True)
    scores={}
    for name in candidates()['both_all_architectures']:
        size,trial=name.split('/')
        data=pd.read_csv(roots[int(size)]/trial/'selection_predictions.csv').sort_values('sha256').reset_index(drop=True)
        for field in ['sha256','patient']:
            if not data[field].equals(expected[field]): raise ValueError('Selection identity mismatch: '+name)
        if not np.array_equal(data.y_true,expected.label): raise ValueError('Selection labels disagree')
        scores[name]=data.probability.to_numpy(float)
    return expected, scores


def source_path(row, data_root):
    return data_root/row.original_split/('NORMAL' if row.y_true==0 else 'PNEUMONIA')/row.filename


def infer(frame, members, data_root, device, output, split):
    # Decode identity before any inference. Preprocessing is the original full-image letterbox.
    paths=[]
    for row in frame.itertuples():
        path=source_path(row,data_root)
        with Image.open(path) as image:
            gray=image.convert('L')
            digest=hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()
        if digest!=row.sha256: raise ValueError('Pixel identity mismatch: '+row.filename)
        paths.append(path)
    result={}
    for size in sorted({m['image_size'] for m in members}):
        tensors=[image_tensor(p,size) for p in paths]
        for member in [m for m in members if m['image_size']==size]:
            model=model_for(member['architecture'])
            model.load_state_dict(torch.load(member['original_path'],map_location='cpu',weights_only=True),strict=True)
            model.to(device).eval(); values=[]
            with torch.inference_mode():
                for start in range(0,len(tensors),32):
                    batch=torch.stack(tensors[start:start+32]).to(device)
                    values.extend(torch.sigmoid(model(batch).flatten()).cpu().numpy().astype(float))
            values=np.asarray(values)
            if len(values)!=len(frame) or not np.isfinite(values).all(): raise ValueError('Invalid inference output')
            result[member['name']]=values
            print('MEMBER_INFERENCE',split,member['name'],len(values),flush=True)
            del model
        del tensors
    raw=np.mean([result[m['name']] for m in members],axis=0)
    member_frame=frame[['sha256','patient','y_true']].copy()
    for name,values in result.items(): member_frame[name]=values
    member_frame.to_csv(output/f'{split}_raw_member_scores.csv',index=False)
    return raw


def run(low,high,data_root,output,protocol,locks):
    start=time.monotonic()
    low,high,data_root,output=map(Path,[low,high,data_root,output])
    output.mkdir(parents=True,exist_ok=True)
    save(output/'protocol.json',protocol)
    torch.set_num_threads(2)
    torch.manual_seed(20261026)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    roots={224:low,320:high}
    frame=aligned_manifest(low,high)
    # Check all original weights without looking at any test probability file.
    for name in candidates()['both_all_architectures']:
        m=member_info(name,roots)
        digest=hashlib.sha256(m['original_path'].read_bytes()).hexdigest()
        if digest!=locks[name]: raise ValueError('Original checkpoint lock mismatch: '+name)
    selected_frame,scores=selection_scores(frame,roots)
    choice=choose(selected_frame.label,selected_frame.patient,scores)
    save(output/'development_choice.json',choice)
    selected_frame.assign(threshold_cv_fold=choice['fold_ids']).to_csv(output/'selection_folds.csv',index=False)
    print('DEVELOPMENT_CHOICE',json.dumps(choice),flush=True)
    members=[member_info(name,roots) for name in choice['selected']['members']]
    calibration=frame[frame.split=='calibration'].rename(columns={'label':'y_true'}).reset_index(drop=True)
    raw=infer(calibration,members,data_root,device,output,'calibration')
    threshold,correct=accuracy_threshold(calibration.y_true,raw)
    config=dict(classes=['NORMAL','PNEUMONIA'],aggregation='mean_raw_sigmoid',threshold=threshold,
                selected_candidate=choice['selected']['candidate'],members=[],
                objective='Maximum overall accuracy on the separate calibration partition',
                threshold_tie_rule='Lowest midpoint threshold among maximum-accuracy candidates',
                score_interpretation='Uncalibrated research score, not a clinical probability',
                status='Exploratory reuse of previously inspected source cohorts')
    for member in members:
        relative='members/'+member['name'].replace('/','_')+'/best_weights.pt'
        dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(member['original_path'],dest)
        config['members'].append(dict(architecture=member['architecture'],image_size=member['image_size'],
                                     checkpoint=relative,sha256=locks[member['name']]))
    # Freeze model, exported weights, and cutoff BEFORE reading old or new test outcomes.
    save(output/'accuracy_config.json',config)
    frozen_sha=hashlib.sha256((output/'accuracy_config.json').read_bytes()).hexdigest()
    calibration=calibration.assign(probability=raw,prediction=(raw>=threshold).astype(int))
    calibration.to_csv(output/'calibration_predictions.csv',index=False)
    save(output/'calibration_fit.json',dict(images=len(calibration),correct=correct,
        metrics=metrics(calibration),threshold=threshold,config_sha256=frozen_sha,
        limitation='In-sample cutoff fitting; not an independent estimate of future accuracy'))
    print('MODEL_FROZEN',json.dumps(dict(candidate=config['selected_candidate'],threshold=threshold,config_sha256=frozen_sha)),flush=True)
    test=frame[frame.split=='test'].rename(columns={'label':'y_true'}).reset_index(drop=True)
    p=infer(test,members,data_root,device,output,'test')
    test=test.assign(probability=p,prediction=(p>=threshold).astype(int))
    test.to_csv(output/'test_predictions.csv',index=False)
    comparison=analyze(high,output,output/'paired_uncertainty',replicates=1000)
    cli=Path(__file__).with_name('predict_accuracy.py')
    shutil.copyfile(cli,output/'predict_accuracy.py')
    checks=[]
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    for row in test.iloc[:2].itertuples():
        process=subprocess.run([sys.executable,str(cli),'--model-dir',str(output),'--image',str(source_path(row,data_root)),'--device','cpu'],capture_output=True,text=True,timeout=600,env=env)
        if process.returncode!=0: raise RuntimeError('Inference CLI failed: '+process.stderr[-2000:])
        answer=json.loads(process.stdout)
        difference=abs(answer['pneumonia_score']-row.probability)
        changed=int(answer['predicted_label']!=config['classes'][row.prediction])
        checks.append(dict(sha256=row.sha256,score_difference=difference,changed_prediction=changed))
    passed=all(c['score_difference']<=1e-4 and c['changed_prediction']==0 for c in checks)
    if hashlib.sha256((output/'accuracy_config.json').read_bytes()).hexdigest()!=frozen_sha:
        raise ValueError('Configuration changed after freeze')
    report=dict(status='completed' if passed else 'CLI numerical verification failed',
                selected_candidate=config['selected_candidate'],members=len(members),
                selection_oof_accuracy=choice['selected']['oof_accuracy'],
                calibration_accuracy=correct/len(calibration),threshold=threshold,
                test_metrics=metrics(test),paired_comparison=comparison,
                cli_verification=dict(passed=passed,images=2,selection='First two ascending pixel hashes',checks=checks,tolerance=1e-4),
                confusion=dict(tn=int(((test.y_true==0)&(test.prediction==0)).sum()),
                               fp=int(((test.y_true==0)&(test.prediction==1)).sum()),
                               fn=int(((test.y_true==1)&(test.prediction==0)).sum()),
                               tp=int(((test.y_true==1)&(test.prediction==1)).sum())),
                environment=dict(device=device,python=sys.version,torch=torch.__version__,numpy=np.__version__,
                                 pandas=pd.__version__,sklearn=sklearn.__version__,
                                 gpu=torch.cuda.get_device_name(0) if device=='cuda' else None),
                runtime_seconds=time.monotonic()-start,
                config_sha256=frozen_sha,training_performed=False,
                limitations=protocol['limitations'])
    save(output/'summary.json',report)
    print('ACCURACY_EXPERIMENT_DONE',json.dumps(report),flush=True)
    if not passed: raise RuntimeError('CLI check failed; frozen results retained')
    return report
