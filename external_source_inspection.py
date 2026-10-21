"""Inspect public mirror documentation before any external model evaluation."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pandas as pd

try:
    import pypdf
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','pypdf'],check=True)
    import pypdf


def inspect():
    base=Path('/kaggle/input')
    agreements=list(base.rglob('Data Use Agreement.pdf'))
    if len(agreements)!=1:
        raise ValueError('Attach only the BDCXR3257 mirror with its supporting documents')
    documents=agreements[0].parent
    out=Path('/kaggle/working/external_source_inspection');out.mkdir(exist_ok=True)
    report={'purpose':'Source and documentation inspection only; no model inference or agreement acceptance',
            'mirror':'https://www.kaggle.com/datasets/wasimulbarirahat/bdcxr3257',
            'mirror_license_display':'Unknown','original_source_doi':'10.6084/m9.figshare.32086362',
            'documents':[],'metadata':[]}
    for path in sorted(documents.glob('*.pdf')):
        reader=pypdf.PdfReader(path)
        record={'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'pages':len(reader.pages),'page_text':[],'extracted_images':[]}
        for index,page in enumerate(reader.pages):
            text=page.extract_text() or ''
            record['page_text'].append(text)
            # Scanned pages may have no extractable text. Preserve their embedded images for review.
            for image_index,image in enumerate(page.images):
                suffix=Path(image.name).suffix or '.png'
                name=f'{path.stem}_page{index+1}_image{image_index+1}{suffix}'
                (out/name).write_bytes(image.data);record['extracted_images'].append(name)
        report['documents'].append(record)
    for path in sorted(base.rglob('metadata.csv')):
        frame=pd.read_csv(path)
        report['metadata'].append({'path':str(path),'rows':len(frame),'columns':list(frame.columns),
                                   'missing_values':frame.isna().sum().to_dict(),
                                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report['image_counts']={name:sum(1 for folder in base.rglob(name) if folder.is_dir()
        for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
        for name in ['Normal','Pneumonia Bacterial','Pneumonia Viral']}
    (out/'inspection.json').write_text(json.dumps(report,indent=2))
    print('DOCUMENT_INSPECTION',json.dumps(report),flush=True)


if __name__=='__main__': inspect()
