"""Prepare an exact-byte GitHub commit payload; this script performs no network write."""
import base64
import hashlib
import json
from pathlib import Path
import zipfile

from package_research import package


def prepare():
    root=Path(__file__).resolve().parent
    archive_path=package(root)
    upserts=[];expected=[]
    with zipfile.ZipFile(archive_path) as archive:
        manifest=json.loads(archive.read('SOURCE_MANIFEST.json'))
        for entry in manifest['files']:
            path=entry['path'];content=archive.read(path)
            if hashlib.sha256(content).hexdigest()!=entry['sha256']:
                raise ValueError('Source manifest mismatch: '+path)
            upserts.append(dict(path=path,content=base64.b64encode(content).decode('ascii'),encoding='base64'))
            # Git blob SHA permits checking every published byte without re-downloading each file.
            blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
            expected.append(dict(path=path,bytes=len(content),git_blob_sha=blob,sha256=entry['sha256']))
        content=archive.read('SOURCE_MANIFEST.json')
        upserts.append(dict(path='SOURCE_MANIFEST.json',content=base64.b64encode(content).decode('ascii'),encoding='base64'))
        expected.append(dict(path='SOURCE_MANIFEST.json',bytes=len(content),
            git_blob_sha=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest(),
            sha256=hashlib.sha256(content).hexdigest()))
    output=root/'artifacts'
    (output/'github_upserts.json').write_text(json.dumps(upserts,separators=(',',':')),encoding='utf-8')
    (output/'github_expected_files.json').write_text(json.dumps(expected,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='Prepared locally; not uploaded',files=len(upserts),
        payload_bytes=(output/'github_upserts.json').stat().st_size,
        requirement='Regenerate after final results/report changes; resolve the actual remote default branch before committing'),indent=2))


if __name__=='__main__':
    prepare()
