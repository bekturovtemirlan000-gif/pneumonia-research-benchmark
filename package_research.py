"""Build a source/results archive using an explicit publication allowlist."""
import hashlib
import json
from pathlib import Path
import zipfile

from audit_reproducibility import audit


ROOT_FILES = ['README.md','MODEL_CARD.md','.gitignore','requirements-local-lock.txt',
              'research_protocol.json','resolution_extension_protocol.json',
              'external_evaluation_protocol.json','research_evidence_notes.json',
              'selection_stability_protocol.json']


def package(root=Path('.')):
    root=Path(root).resolve()
    audit(root)
    files=[root/name for name in ROOT_FILES]
    for pattern in ['*.py','*.ipynb','tests/*.py','results/*.json',
                    'output/pdf/*.pdf','output/figures/**/*.png','output/figures/**/*.pdf']:
        files.extend(root.glob(pattern))
    files=sorted(set(files),key=lambda p:p.relative_to(root).as_posix())
    manifest=[]
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError('Missing or nonregular publication file: '+str(path))
        relative=path.relative_to(root).as_posix()
        if path.suffix=='.ipynb':
            notebook=json.loads(path.read_text(encoding='utf-8'))
            if any(cell.get('outputs') for cell in notebook['cells']):
                raise ValueError('Notebook contains execution outputs requiring review: '+relative)
        payload=path.read_bytes()
        manifest.append(dict(path=relative,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    record=dict(status='Exploratory research source release; not peer reviewed or clinically validated',
                scope='Code, protocols, compact results, figures and draft report',
                excluded='Raw datasets, model weights, local artifacts, credentials and browser screenshots',
                files=manifest)
    out=root/'artifacts';out.mkdir(exist_ok=True)
    target=out/'research_source_bundle.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in manifest:
            archive.write(root/item['path'],item['path'])
        archive.writestr('SOURCE_MANIFEST.json',json.dumps(record,indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None: raise ValueError('Archive CRC failure')
        if len(archive.namelist())!=len(manifest)+1: raise ValueError('Archive file count mismatch')
        for item in manifest:
            if hashlib.sha256(archive.read(item['path'])).hexdigest()!=item['sha256']:
                raise ValueError('Archive payload mismatch: '+item['path'])
    (out/'research_source_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(target),files=len(manifest),bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
    return target


if __name__=='__main__':
    package()
