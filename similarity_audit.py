"""Exploratory cross-partition image-similarity audit. Never selects a model."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn
from research_benchmark import split_cohorts, patient_id


def fingerprint(image):
    gray=image.convert('L')
    small=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    values=dctn(small,type=2,norm='ortho')[:8,:8].flatten()
    bits=values>np.median(values[1:]); bits[0]=False
    value=np.uint64(0)
    for bit in bits: value=(value<<np.uint64(1))|np.uint64(bit)
    return value


def run_similarity_audit():
    out=Path('/kaggle/working/similarity_audit_outputs'); out.mkdir(parents=True,exist_ok=True)
    roots=sorted({p.parent for p in Path('/kaggle/input').rglob('train')
        if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
    if not roots: raise ValueError('Attach source chest X-ray dataset')
    root=roots[0]; rows=[]
    for split in ['train','val','test']:
        for label,cls in enumerate(['NORMAL','PNEUMONIA']):
            for path in sorted((root/split/cls).glob('*')):
                if path.suffix.lower() not in {'.png','.jpeg','.jpg'}: continue
                with Image.open(path) as image:
                    gray=image.convert('L')
                    sha=hashlib.sha256(str(gray.size).encode()+gray.tobytes()).hexdigest()
                    phash=fingerprint(image)
                rows.append(dict(path=str(path),filename=path.name,label=label,original_split=split,
                    patient=patient_id(path.name),sha256=sha,phash=phash))
                if len(rows)%500==0: print('Fingerprint audit:',len(rows),flush=True)
    frames=split_cohorts(pd.DataFrame(rows))
    manifest=pd.concat([f.assign(split=k) for k,f in frames.items()],ignore_index=True)
    manifest.assign(phash=manifest.phash.map(lambda x:f'{int(x):016x}')).to_csv(out/'fingerprints.csv',index=False)
    names=list(frames); all_nearest=[]; candidates=[]
    for i,a in enumerate(names):
        for b in names[i+1:]:
            right=frames[b]; right_hash=right.phash.to_numpy(dtype=np.uint64)
            for left in frames[a].itertuples():
                distances=np.bitwise_count(np.bitwise_xor(np.uint64(left.phash),right_hash))
                index=int(np.argmin(distances)); match=right.iloc[index]
                all_nearest.append(dict(split_a=a,split_b=b,path_a=left.path,path_b=match.path,
                    label_a=left.label,label_b=int(match.label),distance=int(distances[index])))
                for index in np.flatnonzero(distances<=6):
                    match=right.iloc[int(index)]
                    candidates.append(dict(split_a=a,split_b=b,path_a=left.path,path_b=match.path,
                        label_a=left.label,label_b=int(match.label),distance=int(distances[index])))
    pd.DataFrame(all_nearest).to_csv(out/'nearest_cross_partition.csv',index=False)
    fields=['split_a','split_b','path_a','path_b','label_a','label_b','distance']
    pairs=pd.DataFrame(candidates,columns=fields).sort_values(['distance','path_a','path_b'])
    pairs.to_csv(out/'candidate_pairs.csv',index=False)
    # Candidate similarity is not proof of duplicated identity or label validity.
    for page in range((min(len(pairs),40)+7)//8):
        sheet=Image.new('RGB',(900,8*190),'white'); draw=ImageDraw.Draw(sheet)
        for row_index,(_,row) in enumerate(pairs.iloc[page*8:(page+1)*8].iterrows()):
            y=row_index*190
            for column,key in enumerate(['path_a','path_b']):
                with Image.open(row[key]) as image:
                    thumb=ImageOps.pad(image.convert('RGB'),(230,160),color='black')
                sheet.paste(thumb,(column*450,y+25))
                draw.text((column*450+235,y+30),Path(row[key]).name[:32],fill='black')
            draw.text((4,y+4),f'{row.split_a} vs {row.split_b}; pHash distance {row.distance}; labels {row.label_a}/{row.label_b}',fill='black')
        sheet.save(out/f'candidate_review_{page+1}.png')
    summary=dict(status='Exploratory similarity screening, not verified duplicate detection',
        retained_images=len(manifest),candidate_pairs=len(pairs),
        candidate_test_pairs=int(((pairs.split_a=='test')|(pairs.split_b=='test')).sum()),
        candidate_rule='64-bit low-frequency DCT perceptual hash, DC bit zero, Hamming distance <=6',
        limitations=['Common anatomy can produce false matches','Transforms may evade this screening',
                    'Only first 40 closest pairs have contact sheets','No cohort exclusions or model changes made from this audit'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',json.dumps(summary),flush=True)


if __name__=='__main__':
    run_similarity_audit()
