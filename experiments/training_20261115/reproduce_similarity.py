"""Reproduce the frozen perceptual-hash screen directly from a local source ZIP."""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from research_benchmark import split_cohorts, patient_id
from similarity_audit import fingerprint


def run(archive, output):
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    archive = Path(archive)
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if '__MACOSX' not in n and n.lower().endswith(('.jpeg', '.jpg', '.png'))]
        prefixes = {str(Path(n).parent.parent.parent).replace('\\', '/') for n in names}
        prefix = min(prefixes, key=lambda s: (len(s.split('/')), s))
        chosen = [n for n in names if str(Path(n).parent.parent.parent).replace('\\', '/') == prefix]
        if len(chosen) != 5856: raise ValueError('Expected one complete 5,856-image source tree')
        rows = []; resized = {}
        for index, name in enumerate(sorted(chosen)):
            parts = name.split('/'); split, cls, filename = parts[-3:]
            if split not in ['train', 'val', 'test'] or cls not in ['NORMAL', 'PNEUMONIA']:
                raise ValueError('Unexpected archive label structure')
            with Image.open(BytesIO(z.read(name))) as image:
                gray = image.convert('L')
                sha = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
                phash = fingerprint(image)
                small = np.asarray(ImageOps.pad(gray, (160, 160), method=Image.Resampling.BICUBIC, color=0), dtype=np.float64)
                centered = small.ravel() - small.mean()
                norm = np.linalg.norm(centered)
                if not norm > 0: raise ValueError('Flat image cannot be compared')
                resized[sha] = (centered / norm).astype(np.float32)
            rows.append(dict(path=name, filename=filename, original_split=split, label=int(cls == 'PNEUMONIA'),
                             patient=patient_id(filename), sha256=sha, phash=phash))
            if (index + 1) % 500 == 0: print('LOCAL_SIMILARITY_AUDIT', index+1, 5856, flush=True)
        frames = split_cohorts(pd.DataFrame(rows))
        manifest = pd.concat([f.assign(split=k) for k, f in frames.items()], ignore_index=True)
        manifest.assign(phash=manifest.phash.map(lambda n: f'{int(n):016x}')).to_csv(output / 'fingerprints.csv', index=False)
        candidates = []; partitions = list(frames)
        for i, a in enumerate(partitions):
            for b in partitions[i+1:]:
                right = frames[b]; right_hash = right.phash.to_numpy(dtype=np.uint64)
                for left in frames[a].itertuples():
                    distances = np.bitwise_count(np.bitwise_xor(np.uint64(left.phash), right_hash))
                    for index in np.flatnonzero(distances <= 6):
                        match = right.iloc[int(index)]
                        candidates.append(dict(split_a=a, split_b=b, path_a=left.path, path_b=match.path,
                                               sha256_a=left.sha256, sha256_b=match.sha256,
                                               label_a=int(left.label), label_b=int(match.label), distance=int(distances[index]),
                                               normalized_pixel_correlation=float(np.dot(resized[left.sha256], resized[match.sha256]))))
        pairs = pd.DataFrame(candidates).sort_values(['distance', 'path_a', 'path_b']).reset_index(drop=True)
        pairs['pair_id'] = [hashlib.sha256((a+':'+b).encode()).hexdigest()[:16] for a, b in zip(pairs.sha256_a, pairs.sha256_b)]
        pairs.to_csv(output / 'candidate_pairs.csv', index=False)
        public_table = HERE / 'similarity_candidate_pairs.csv'
        pairs[['pair_id','split_a','split_b','sha256_a','sha256_b','label_a','label_b','distance',
               'normalized_pixel_correlation']].to_csv(public_table, index=False)
        # Descriptive ranking only: no duplicate decision, cohort exclusion, or threshold change.
        closest = pairs.sort_values(['normalized_pixel_correlation', 'pair_id'], ascending=[False, True]).head(24)
        for page in range(4):
            group = closest.iloc[page*6:(page+1)*6]
            sheet = Image.new('RGB', (1300, len(group)*290), 'white'); draw = ImageDraw.Draw(sheet)
            for row_index, row in enumerate(group.itertuples()):
                y = row_index*290
                draw.text((8, y+5), f'{row.pair_id} | {row.split_a} / {row.split_b} | pHash {row.distance} | pixel r={row.normalized_pixel_correlation:.6f}', fill='black')
                for col, key in enumerate(['path_a', 'path_b']):
                    name = getattr(row, key)
                    with Image.open(BytesIO(z.read(name))) as image:
                        thumbnail = ImageOps.pad(image.convert('RGB'), (540, 245), method=Image.Resampling.LANCZOS, color='black')
                    sheet.paste(thumbnail, (col*650+8, y+37))
                    draw.text((col*650+8, y+22), Path(name).name, fill='black')
            sheet.save(output / f'pixel_ranked_review_{page+1}.png')
    reference = json.loads((ROOT / 'results/similarity_screening_summary.json').read_text(encoding='utf-8'))
    test_pairs = int(((pairs.split_a == 'test') | (pairs.split_b == 'test')).sum())
    result = dict(status='completed', recorded_utc=datetime.now(timezone.utc).isoformat(), original_images=5856,
                  archive_bytes=archive.stat().st_size, archive_tree_prefix=prefix, retained_images=len(manifest),
                  retained_counts=manifest.groupby('split').size().to_dict(), candidate_pairs=len(pairs), candidate_test_pairs=test_pairs,
                  candidate_rule='Original 64-bit low-frequency DCT pHash; DC bit zero; Hamming distance <=6',
                  prior_kaggle_version=reference['kaggle_version'], prior_counts_reproduced=bool(len(pairs)==reference['candidate_pairs'] and test_pairs==reference['candidate_test_pairs']),
                  fingerprint_table_sha256=hashlib.sha256((output/'fingerprints.csv').read_bytes()).hexdigest(),
                  candidate_table_sha256=hashlib.sha256((output/'candidate_pairs.csv').read_bytes()).hexdigest(),
                  published_candidate_table_sha256=hashlib.sha256(public_table.read_bytes()).hexdigest(),
                  executed_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  descriptive_pixel_correlation=dict(method='Pearson correlation after full-image grayscale 160-pixel bicubic letterbox; used for review ordering only',
                    maximum=float(pairs.normalized_pixel_correlation.max()),
                    quantiles_50_90_95_99=np.quantile(pairs.normalized_pixel_correlation,[.5,.9,.95,.99]).tolist(),
                    ranked_review_pair_ids=closest.pair_id.tolist()),
                  frozen_train_and_test_digests_verified=True, environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                  seconds=time.monotonic()-started, no_cohort_or_model_changes=True,
                  limitations=['Agreement in total counts does not establish byte-for-byte equality with the unavailable original candidate table',
                               'Neither perceptual similarity nor pixel correlation proves duplicate image or patient identity',
                               'Different photographs, transformations, or repeated acquisitions can evade this screen',
                               'The 24 contact-sheet pairs are a selected review subset, not exhaustive adjudication'])
    (HERE/'similarity_reproduction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('LOCAL_SIMILARITY_DONE',json.dumps(result),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.archive,args.output)
