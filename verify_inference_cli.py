"""Smoke-test the actual exported inference CLI on a deterministic six-image subset."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
from PIL import Image


def verify(model_dir, data_root, source_dir, lock, output):
    model_dir, data_root, source_dir, output = map(Path, (model_dir, data_root, source_dir, output))
    config = json.loads((model_dir / 'inference_config.json').read_text(encoding='utf-8'))
    if config != lock['configuration']:
        raise ValueError('Selected configuration differs from frozen model lock')
    source_hash = hashlib.sha256((source_dir / 'predict.py').read_text(encoding='utf-8').encode()).hexdigest()
    if source_hash != lock['inference_source_sha256_lf']:
        raise ValueError('Inference source changed')
    for checkpoint in lock['checkpoints']:
        if hashlib.sha256((model_dir / checkpoint['checkpoint']).read_bytes()).hexdigest() != checkpoint['sha256']:
            raise ValueError('Checkpoint hash differs from model lock')
    frame = pd.read_csv(model_dir / 'test_predictions.csv', float_precision='round_trip')
    if len(frame) != 618 or frame.sha256.duplicated().any():
        raise ValueError('Unexpected test manifest')
    selected = frame.sort_values('sha256').head(6)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selected_hashes.json').write_text(json.dumps(selected.sha256.tolist()), encoding='utf-8')
    records = []
    environment = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    for row in selected.itertuples():
        image = data_root / row.original_split / config['classes'][int(row.y_true)] / row.filename
        with Image.open(image) as original:
            gray = original.convert('L')
            digest = hashlib.sha256(str(gray.size).encode() + gray.tobytes()).hexdigest()
        if digest != row.sha256:
            raise ValueError('Image identity mismatch')
        command = [sys.executable, str(source_dir / 'predict.py'), '--model-dir', str(model_dir), '--image', str(image), '--device', 'cpu']
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180, env=environment)
        prediction = json.loads(result.stdout)
        difference = abs(prediction['pneumonia_score'] - row.probability)
        changed = prediction['predicted_label'] != config['classes'][int(row.prediction)]
        records.append(dict(sha256=row.sha256, saved_score=row.probability,
                            cli_score=prediction['pneumonia_score'], absolute_score_difference=difference,
                            changed_classification=changed, exit_code=result.returncode))
        print('CLI image completed', len(records), flush=True)
    maximum = max(row['absolute_score_difference'] for row in records)
    changed = sum(row['changed_classification'] for row in records)
    passed = maximum <= 1e-4 and changed == 0
    report = dict(status='passed' if passed else 'failed', images=len(records),
                  selection='First six test images by ascending decoded-pixel SHA-256',
                  max_absolute_score_difference=maximum, changed_classifications=changed,
                  score_tolerance=1e-4, inference_source_sha256_lf=source_hash,
                  checkpoint_hashes_verified=lock['checkpoints'], records=records,
                  scope='Actual predict.py subprocess smoke test; not full-cohort or external validation')
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('CLI_SMOKE', json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError('Inference CLI discrepancy; preserve result and investigate')
    return report
