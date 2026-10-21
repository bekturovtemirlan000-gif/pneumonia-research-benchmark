"""Prespecified multi-architecture pneumonia benchmark. Research use only."""
from pathlib import Path
import hashlib, json, random, re, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, log_loss
import sklearn
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

ARCHITECTURES = ['efficientnet_b0', 'densenet121', 'resnet18']
SEEDS = [42, 123, 2026]
FROZEN_VALIDATION_GROUPS = set(['im:0115', 'im:0141', 'im:0149', 'im:0152', 'im:0164', 'im:0182', 'im:0189', 'im:0195', 'im:0199', 'im:0203', 'im:0209', 'im:0221', 'im:0225', 'im:0234', 'im:0239', 'im:0244', 'im:0249', 'im:0261', 'im:0269', 'im:0276', 'im:0278', 'im:0285', 'im:0289', 'im:0295', 'im:0302', 'im:0307', 'im:0312', 'im:0317', 'im:0325', 'im:0333', 'im:0341', 'im:0349', 'im:0357', 'im:0359', 'im:0367', 'im:0379', 'im:0385', 'im:0391', 'im:0394', 'im:0399', 'im:0409', 'im:0416', 'im:0420', 'im:0430', 'im:0435', 'im:0444', 'im:0449', 'im:0455', 'im:0459', 'im:0461', 'im:0466', 'im:0479', 'im:0485', 'im:0487', 'im:0494', 'im:0499', 'im:0510', 'im:0514', 'im:0516', 'im:0521', 'im:0525', 'im:0526', 'im:0532', 'im:0537', 'im:0545', 'im:0552', 'im:0559', 'im:0563', 'im:0570', 'im:0574', 'im:0583', 'im:0596', 'im:0598', 'im:0607', 'im:0609', 'im:0613', 'im:0615', 'im:0622', 'im:0623', 'im:0629', 'im:0636', 'im:0637', 'im:0652', 'im:0656', 'im:0662', 'im:0663', 'im:0665', 'im:0671', 'im:0675', 'im:0681', 'im:0687', 'im:0691', 'im:0697', 'im:0698', 'im:0707', 'im:0717', 'im:0727', 'im:0732', 'im:0736', 'im:0742', 'im:0746', 'im:0755', 'normal2:0388', 'normal2:0389', 'normal2:0395', 'normal2:0402', 'normal2:0407', 'normal2:0423', 'normal2:0429', 'normal2:0433', 'normal2:0435', 'normal2:0447', 'normal2:0454', 'normal2:0458', 'normal2:0466', 'normal2:0479', 'normal2:0485', 'normal2:0487', 'normal2:0488', 'normal2:0502', 'normal2:0503', 'normal2:0520', 'normal2:0525', 'normal2:0530', 'normal2:0539', 'normal2:0553', 'normal2:0555', 'normal2:0557', 'normal2:0564', 'normal2:0573', 'normal2:0576', 'normal2:0578', 'normal2:0585', 'normal2:0599', 'normal2:0603', 'normal2:0611', 'normal2:0617', 'normal2:0627', 'normal2:0634', 'normal2:0635', 'normal2:0641', 'normal2:0642', 'normal2:0650', 'normal2:0661', 'normal2:0664', 'normal2:0668', 'normal2:0672', 'normal2:0675', 'normal2:0683', 'normal2:0692', 'normal2:0702', 'normal2:0736', 'normal2:0741', 'normal2:0765', 'normal2:0790', 'normal2:0797', 'normal2:0807', 'normal2:0812', 'normal2:0825', 'normal2:0829', 'normal2:0834', 'normal2:0842', 'normal2:0849', 'normal2:0853', 'normal2:0854', 'normal2:0862', 'normal2:0865', 'normal2:0875', 'normal2:0886', 'normal2:0887', 'normal2:0897', 'normal2:0906', 'normal2:0908', 'normal2:0911', 'normal2:0924', 'normal2:0929', 'normal2:0934', 'normal2:0937', 'normal2:0944', 'normal2:0947', 'normal2:0948', 'normal2:0956', 'normal2:0960', 'normal2:0978', 'normal2:0980', 'normal2:0988', 'normal2:0997', 'normal2:0999', 'normal2:1001', 'normal2:1015', 'normal2:1020', 'normal2:1025', 'normal2:1039', 'normal2:1048', 'normal2:1051', 'normal2:1055', 'normal2:1060', 'normal2:1090', 'normal2:1098', 'normal2:1103', 'normal2:1114', 'normal2:1122', 'normal2:1127', 'normal2:1132', 'normal2:1142', 'normal2:1149', 'normal2:1156', 'normal2:1161', 'normal2:1164', 'normal2:1167', 'normal2:1175', 'normal2:1181', 'normal2:1188', 'normal2:1196', 'normal2:1203', 'normal2:1214', 'normal2:1221', 'normal2:1223', 'normal2:1231', 'normal2:1237', 'normal2:1245', 'normal2:1247', 'normal2:1261', 'normal2:1270', 'normal2:1276', 'normal2:1281', 'normal2:1285', 'normal2:1293', 'normal2:1294', 'normal2:1300', 'normal2:1307', 'normal2:1318', 'normal2:1319', 'normal2:1329', 'normal2:1332', 'normal2:1337', 'normal2:1345', 'normal2:1351', 'normal2:1362', 'normal2:1365', 'normal2:1401', 'normal2:1427', 'normal2:1436', 'person:1000', 'person:1003', 'person:1004', 'person:1007', 'person:1014', 'person:1017', 'person:1023', 'person:1033', 'person:1041', 'person:1046', 'person:1055', 'person:1057', 'person:106', 'person:1062', 'person:1065', 'person:1068', 'person:1069', 'person:1074', 'person:1083', 'person:1090', 'person:1095', 'person:1102', 'person:1107', 'person:1110', 'person:1112', 'person:1124', 'person:1128', 'person:1130', 'person:1143', 'person:1146', 'person:1152', 'person:1160', 'person:1162', 'person:1165', 'person:1171', 'person:1173', 'person:1185', 'person:1193', 'person:1196', 'person:1197', 'person:1204', 'person:1212', 'person:1216', 'person:1219', 'person:1222', 'person:1225', 'person:1230', 'person:1236', 'person:1243', 'person:1244', 'person:1249', 'person:1256', 'person:1264', 'person:1270', 'person:1275', 'person:1281', 'person:1282', 'person:1296', 'person:1300', 'person:1304', 'person:1305', 'person:1315', 'person:1321', 'person:1322', 'person:1328', 'person:1340', 'person:1348', 'person:1356', 'person:1358', 'person:1363', 'person:1369', 'person:1375', 'person:1378', 'person:1379', 'person:1385', 'person:1390', 'person:1397', 'person:1400', 'person:1402', 'person:1407', 'person:1413', 'person:1423', 'person:1427', 'person:1428', 'person:1439', 'person:1442', 'person:1452', 'person:1455', 'person:1458', 'person:1466', 'person:1471', 'person:1475', 'person:1477', 'person:1486', 'person:1493', 'person:1497', 'person:1504', 'person:1513', 'person:1519', 'person:1522', 'person:1526', 'person:1536', 'person:1542', 'person:1545', 'person:1552', 'person:1562', 'person:1565', 'person:1566', 'person:1571', 'person:1581', 'person:1588', 'person:1589', 'person:1599', 'person:1604', 'person:1611', 'person:1630', 'person:1677', 'person:1683', 'person:1690', 'person:1695', 'person:1702', 'person:1705', 'person:1710', 'person:1718', 'person:1724', 'person:1729', 'person:1735', 'person:1748', 'person:1749', 'person:1757', 'person:1763', 'person:1799', 'person:1810', 'person:1830', 'person:1847', 'person:1852', 'person:1858', 'person:1859', 'person:1867', 'person:1884', 'person:1886', 'person:1905', 'person:1918', 'person:1923', 'person:1930', 'person:1932', 'person:1936', 'person:1941', 'person:1946', 'person:2', 'person:261', 'person:264', 'person:27', 'person:272', 'person:279', 'person:290', 'person:291', 'person:298', 'person:300', 'person:304', 'person:309', 'person:312', 'person:319', 'person:320', 'person:321', 'person:326', 'person:336', 'person:340', 'person:349', 'person:352', 'person:357', 'person:367', 'person:369', 'person:372', 'person:375', 'person:377', 'person:382', 'person:389', 'person:390', 'person:400', 'person:401', 'person:412', 'person:419', 'person:420', 'person:425', 'person:431', 'person:438', 'person:445', 'person:451', 'person:457', 'person:459', 'person:465', 'person:470', 'person:478', 'person:480', 'person:484', 'person:489', 'person:494', 'person:495', 'person:502', 'person:508', 'person:510', 'person:514', 'person:521', 'person:524', 'person:534', 'person:541', 'person:543', 'person:550', 'person:553', 'person:559', 'person:562', 'person:566', 'person:567', 'person:574', 'person:581', 'person:587', 'person:593', 'person:598', 'person:601', 'person:610', 'person:614', 'person:616', 'person:621', 'person:627', 'person:632', 'person:635', 'person:641', 'person:648', 'person:655', 'person:657', 'person:665', 'person:674', 'person:68', 'person:685', 'person:687', 'person:700', 'person:705', 'person:706', 'person:711', 'person:718', 'person:724', 'person:731', 'person:740', 'person:746', 'person:747', 'person:755', 'person:758', 'person:763', 'person:767', 'person:772', 'person:774', 'person:775', 'person:786', 'person:787', 'person:789', 'person:796', 'person:802', 'person:809', 'person:811', 'person:817', 'person:831', 'person:835', 'person:837', 'person:838', 'person:851', 'person:852', 'person:862', 'person:871', 'person:875', 'person:876', 'person:884', 'person:890', 'person:900', 'person:901', 'person:908', 'person:911', 'person:916', 'person:920', 'person:923', 'person:930', 'person:932', 'person:933', 'person:941', 'person:947', 'person:949', 'person:959', 'person:966', 'person:970', 'person:976', 'person:978', 'person:982', 'person:985', 'person:993', 'person:995'])
EXPECTED_COHORT_HASHES = {'train': '293c6127291992894864b58fa85e1d7438d9b09544c800d8503fd61c9a8c8b6d', 'test': 'c0de6a4a18c8c295d85551b5b8602410e2e26d173b8ab9f141e82cd6fdca9378'}
MEAN, STD = [.485, .456, .406], [.229, .224, .225]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_model(architecture, pretrained=False):
    if architecture == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1)
        head = m.classifier
    elif architecture == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        head = m.classifier
    elif architecture == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        head = m.fc
    else:
        raise ValueError('Unsupported architecture: ' + architecture)
    return m, head


def patient_id(name):
    for pattern, prefix in [(r'person(\d+)_','person'), (r'NORMAL2-IM-(\d+)-','normal2'), (r'IM-(\d+)-','im')]:
        match = re.match(pattern, name, re.I)
        if match: return prefix + ':' + match.group(1)
    raise ValueError('Unrecognized patient filename: ' + name)


def split_cohorts(df):
    if df.groupby('sha256').label.nunique().max() > 1:
        raise ValueError('Conflicting labels for identical decoded images')
    test = df[df.original_split == 'test'].drop_duplicates('sha256').copy()
    dev = df[df.original_split != 'test'].copy()
    dev = dev[~dev.patient.isin(test.patient) & ~dev.sha256.isin(test.sha256)]
    dev = dev.sort_values(['original_split','label','path']).drop_duplicates('sha256').reset_index(drop=True)
    held = dev[dev.patient.isin(FROZEN_VALIDATION_GROUPS)].copy()
    train = dev[~dev.patient.isin(FROZEN_VALIDATION_GROUPS)].copy()
    if held.groupby('patient').label.nunique().max() != 1:
        raise ValueError('Mixed-label patient proxy needs manual grouping review')
    selection_groups = set()
    for label in [0, 1]:
        groups = sorted(held[held.label == label].patient.unique(),
                        key=lambda g: hashlib.sha256(('20261021:' + g).encode()).hexdigest())
        selection_groups.update(groups[::2])
    frames = {'train':train, 'selection':held[held.patient.isin(selection_groups)].copy(),
              'calibration':held[~held.patient.isin(selection_groups)].copy(), 'test':test}
    for name in ['train', 'test']:
        digest = hashlib.sha256('\n'.join(sorted(frames[name].sha256)).encode()).hexdigest()
        if digest != EXPECTED_COHORT_HASHES[name]:
            raise ValueError('Source cohort differs from frozen audited cohort: ' + name)
    names = list(frames)
    for i, name in enumerate(names):
        frame = frames[name]
        if set(frame.label) != {0,1}: raise ValueError('Both classes required in ' + name)
        for other in names[i+1:]:
            for key in ['patient','sha256']:
                if set(frame[key]) & set(frames[other][key]): raise ValueError('Partition leakage: ' + key)
    return {k:v.reset_index(drop=True) for k,v in frames.items()}


def audit_images(root, cache, out):
    rows=[]
    cache.mkdir(parents=True, exist_ok=True)
    for split in ['train','val','test']:
        for label, cls in enumerate(['NORMAL','PNEUMONIA']):
            for p in sorted((root/split/cls).glob('*')):
                if p.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
                with Image.open(p) as im:
                    gray=im.convert('L')
                    digest=hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()
                    resized=ImageOps.pad(im.convert('RGB'),(320,320),method=Image.Resampling.BICUBIC,color=0)
                    cached=cache/(hashlib.sha256(str(p.relative_to(root)).encode()).hexdigest()+'.png')
                    resized.save(cached)
                rows.append(dict(path=str(p),filename=p.name,label=label,original_split=split,
                                 patient=patient_id(p.name),sha256=digest,cache_path=str(cached)))
                if len(rows)%500==0: print('Audited images:',len(rows),flush=True)
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError('No source images found')
    df.drop(columns='cache_path').to_csv(out/'raw_inventory.csv',index=False)
    frames=split_cohorts(df)
    manifest=pd.concat([f.assign(split=k) for k,f in frames.items()])
    manifest.drop(columns='cache_path').to_csv(out/'split_manifest.csv',index=False)
    audit={'raw_images':len(df),'retained_images':len(manifest),
           'counts':{k:{'images':len(f),'groups':int(f.patient.nunique()),'normal':int((f.label==0).sum()),'pneumonia':int((f.label==1).sum())} for k,f in frames.items()},
           'grouping':'Filename-derived proxies, not verified clinical patient IDs',
           'cache':'Lossless PNG of RGB bicubic full-image letterbox 320; same preprocessing as prior runs'}
    (out/'data_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)
    return frames


class Radiographs(Dataset):
    def __init__(self, frame, training=False):
        self.frame=frame.reset_index(drop=True)
        steps=[transforms.RandomAffine(7,translate=(.03,.03),scale=(.95,1.05))] if training else []
        self.transform=transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        with Image.open(row.cache_path) as im: x=self.transform(im.convert('RGB'))
        return x,torch.tensor(row.label,dtype=torch.float32)


def seed_worker(worker_id):
    s=torch.initial_seed()%2**32; np.random.seed(s); random.seed(s)


def loader(frame, seed, training=False):
    return DataLoader(Radiographs(frame,training),batch_size=32,shuffle=training,num_workers=2,
                      pin_memory=True,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed))


@torch.inference_mode()
def probabilities(model, data, device):
    model.eval(); values=[]
    for x,_ in data:
        values.extend(torch.sigmoid(model(x.to(device)).flatten().float()).cpu().numpy())
    result=np.asarray(values,dtype=float)
    if not np.isfinite(result).all(): raise ValueError('Nonfinite inference scores')
    return result


def threshold_for(y,p):
    fpr,tpr,thresholds=roc_curve(y,p)
    ok=np.isfinite(thresholds)
    return float(thresholds[ok][np.argmax((tpr-fpr)[ok])])


def temperature_for(y,p):
    z=logit(np.clip(p,1e-7,1-1e-7))
    def loss(log_temperature):
        scaled=z/np.exp(log_temperature)
        return float(np.mean(np.logaddexp(0,scaled)-np.asarray(y)*scaled))
    fit=minimize_scalar(loss,bounds=(-3,3),method='bounded')
    if not fit.success: raise ValueError('Temperature fit failed')
    return float(np.exp(fit.x))


def calibrated(p,temperature):
    return expit(logit(np.clip(p,1e-7,1-1e-7))/temperature)


def metrics(y,p,threshold):
    y=np.asarray(y,dtype=int); pred=p>=threshold
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'auroc':float(roc_auc_score(y,p)),'average_precision':float(average_precision_score(y,p)),
            'accuracy':float(np.mean(y==pred)),'sensitivity':float(tp/(tp+fn)),
            'specificity':float(tn/(tn+fp)),'brier_score':float(np.mean((y-p)**2)),
            'log_loss':float(log_loss(y,p,labels=[0,1])),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}


def save_predictions(path,frame,p,cutoff):
    frame.drop(columns='cache_path').assign(y_true=frame.label,probability=p,prediction=(p>=cutoff).astype(int)).to_csv(path,index=False)


def run():
    started=time.monotonic(); deadline=started+180*60
    out=Path('/kaggle/working/pneumonia_resolution320_outputs'); out.mkdir(parents=True,exist_ok=True)
    cache=Path('/kaggle/temp/pneumonia_resolution320_cache')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('Enable Kaggle GPU before running')
    seed_all(42)
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    roots=sorted({p.parent for p in Path('/kaggle/input').rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
    if not roots: raise RuntimeError('Attach paultimothymooney/chest-xray-pneumonia')
    frames=audit_images(roots[0],cache,out)
    data={k:loader(v,42) for k,v in frames.items() if k!='train'}
    # Rotate architecture order across seeds to reduce systematic budget-order effects.
    trials=[(ARCHITECTURES[(i+j)%3],seed) for i,seed in enumerate(SEEDS) for j in range(3)]
    protocol={'architectures':ARCHITECTURES,'seeds':SEEDS,'trial_order':trials,'epochs_max':60,'patience':8,
              'selection':'Architecture with highest mean selection AUROC over all three seeds; average its seed probabilities',
              'calibration':'Separate grouped partition; positive scalar temperature followed by calibration Youden threshold',
              'test_status':'Exploratory previously inspected original test partition; never used for selection',
              'evaluation_precision':'float32','training_budget_minutes':180,'image_size':320,'batch_size':32,
              'learning_rates':{'backbone':.0001,'head':.001},'weight_decay':.0001}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]; saved={}
    for index,(arch,seed) in enumerate(trials):
        if deadline-time.monotonic()<120: raise RuntimeError('Insufficient time to complete all prespecified trials; partial outputs retained')
        trial_deadline=time.monotonic()+(deadline-time.monotonic())/(len(trials)-index)
        trial_id=f'{arch}_seed_{seed}'; dest=out/trial_id; dest.mkdir(exist_ok=True)
        seed_all(seed); train_data=loader(frames['train'],seed,True)
        model,head=make_model(arch,True); model.to(device)
        head_ids={id(p) for p in head.parameters()}
        optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if id(p) not in head_ids],'lr':1e-4},
                                    {'params':head.parameters(),'lr':1e-3}],weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-7)
        ratio=float((frames['train'].label==0).sum()/(frames['train'].label==1).sum())
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio],device=device))
        scaler=torch.amp.GradScaler('cuda'); history=[]; best=-1.; stale=0; best_epoch=None
        trial_start=time.monotonic(); stop_reason='epoch_limit'
        print('START TRIAL',trial_id,flush=True)
        for epoch in range(60):
            if time.monotonic()>=trial_deadline: stop_reason='time_budget'; break
            model.train(); total=0.; seen=0
            for x,y in train_data:
                if time.monotonic()>=trial_deadline: break
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'): loss=criterion(model(x).flatten(),y)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss in '+trial_id)
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                total+=float(loss.item())*len(y); seen+=len(y)
            if not seen: stop_reason='time_budget'; break
            p=probabilities(model,data['selection'],device); auc=float(roc_auc_score(frames['selection'].label,p))
            row={'epoch':epoch+1,'train_loss':total/seen,'train_images_seen':seen,'complete_epoch':seen==len(frames['train']),
                 'selection_auroc':auc,'backbone_lr':optimizer.param_groups[0]['lr'],'head_lr':optimizer.param_groups[1]['lr']}
            history.append(row); pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
            scheduler.step(auc)
            print(trial_id,json.dumps(row),flush=True)
            if auc>best+1e-5:
                best=auc; best_epoch=epoch+1; stale=0; torch.save(model.state_dict(),dest/'best_weights.pt')
            else: stale+=1
            if stale>=8: stop_reason='early_stopping'; break
        if best_epoch is None: raise RuntimeError('No trained checkpoint for '+trial_id)
        model.load_state_dict(torch.load(dest/'best_weights.pt',map_location=device,weights_only=True))
        ps=probabilities(model,data['selection'],device); pc=probabilities(model,data['calibration'],device)
        temperature=temperature_for(frames['calibration'].label,pc)
        cutoff=threshold_for(frames['calibration'].label,calibrated(pc,temperature))
        raw_cutoff=threshold_for(frames['calibration'].label,pc)
        rec={'trial':trial_id,'architecture':arch,'seed':seed,'best_epoch':best_epoch,'epochs_completed':len(history),
             'selection_auroc':float(roc_auc_score(frames['selection'].label,ps)), 'temperature':temperature,
             'threshold':cutoff,'raw_threshold':raw_cutoff,'stop_reason':stop_reason,
             'elapsed_seconds':time.monotonic()-trial_start,'parameter_count':sum(p.numel() for p in model.parameters())}
        records.append(rec); saved[trial_id]={'selection':ps,'calibration':pc}
        save_predictions(dest/'selection_predictions.csv',frames['selection'],ps,.5)
        save_predictions(dest/'calibration_predictions.csv',frames['calibration'],calibrated(pc,temperature),cutoff)
        (dest/'development_summary.json').write_text(json.dumps(rec,indent=2))
        pd.DataFrame(records).to_csv(out/'development_trials.csv',index=False)
        del model,head,optimizer,scheduler,scaler; torch.cuda.empty_cache()
        print('FINISHED TRIAL',trial_id,flush=True)
    table=pd.DataFrame(records)
    if any(len(table[table.architecture==a])!=3 for a in ARCHITECTURES): raise RuntimeError('Incomplete architecture comparison')
    means=table.groupby('architecture',sort=True).selection_auroc.mean()
    selected_arch=str(means.idxmax()); selected_trials=table[table.architecture==selected_arch].trial.tolist()
    pc=np.mean([saved[t]['calibration'] for t in selected_trials],axis=0)
    temp=temperature_for(frames['calibration'].label,pc); cutoff=threshold_for(frames['calibration'].label,calibrated(pc,temp))
    config={'image_size':320,'architecture':selected_arch,'selected_checkpoints':[t+'/best_weights.pt' for t in selected_trials],
            'aggregation':'arithmetic mean of sigmoid probabilities, then positive temperature scaling',
            'temperature':temp,'threshold':cutoff,'classes':['NORMAL','PNEUMONIA'],
            'preprocessing':'RGB; bicubic letterbox 320; ToTensor; ImageNet mean/std',
            'selection_architecture_mean_auroc':means.to_dict(),'research_status':'Exploratory; external confirmation required'}
    (out/'inference_config.json').write_text(json.dumps(config,indent=2))
    print('FROZEN SELECTION',json.dumps(config),flush=True)
    # Only now evaluate the prespecified trials on test. No choices depend on these results.
    tests={}; summaries=[]
    for rec in records:
        model,_=make_model(rec['architecture']); model.to(device)
        model.load_state_dict(torch.load(out/rec['trial']/'best_weights.pt',map_location=device,weights_only=True))
        p=probabilities(model,data['test'],device); tests[rec['trial']]=p
        q=calibrated(p,rec['temperature'])
        save_predictions(out/rec['trial']/'test_predictions_raw.csv',frames['test'],p,rec['raw_threshold'])
        save_predictions(out/rec['trial']/'test_predictions.csv',frames['test'],q,rec['threshold'])
        summaries.append(dict(rec,raw_test=metrics(frames['test'].label,p,rec['raw_threshold']),
                              calibrated_test=metrics(frames['test'].label,q,rec['threshold']),
                              raw_test_at_0_5=metrics(frames['test'].label,p,.5)))
        del model; torch.cuda.empty_cache()
    raw=np.mean([tests[t] for t in selected_trials],axis=0); final=calibrated(raw,temp)
    save_predictions(out/'test_predictions.csv',frames['test'],final,cutoff)
    save_predictions(out/'calibration_predictions.csv',frames['calibration'],calibrated(pc,temp),cutoff)
    ps=np.mean([saved[t]['selection'] for t in selected_trials],axis=0)
    save_predictions(out/'selection_predictions.csv',frames['selection'],calibrated(ps,temp),cutoff)
    result={'threshold':cutoff,'temperature':temp,'test':metrics(frames['test'].label,final,cutoff),
            'raw_test_at_0_5':metrics(frames['test'].label,raw,.5),'selected_architecture':selected_arch,
            'runtime_seconds':time.monotonic()-started,'test_status':'Exploratory, previously inspected source test set'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    (out/'all_trial_results.json').write_text(json.dumps(summaries,indent=2))
    environment={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
                 'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'gpu':torch.cuda.get_device_name(0)}
    (out/'environment.json').write_text(json.dumps(environment,indent=2))
    print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    run()
