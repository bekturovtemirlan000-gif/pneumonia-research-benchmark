"""Local engineering checks; no training claims and no test-score access."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from model import make_model, image_tensor, load_config
from selection import internal_split
from research_benchmark import split_cohorts

torch.set_num_threads(2)
started = time.monotonic()
records = []
for architecture in ['efficientnet_b0', 'efficientnet_v2_s']:
    model, head = make_model(architecture)
    model.train()
    x = torch.randn(2, 3, 64, 64)
    target = torch.tensor([.025, .975])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.classifier[1].weight.detach().clone()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), target)
    loss.backward(); optimizer.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.classifier[1].weight)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 3, 320, 320))
    assert prediction.shape == (1, 1) and torch.isfinite(prediction).all()
    records.append(dict(architecture=architecture, cpu_gradient_update=True, full_resolution_forward=True))
    del model, head, optimizer

old = pd.read_csv(ROOT / 'artifacts/baseline_v1/pneumonia_outputs/split_manifest.csv')
frames = split_cohorts(old.drop(columns='split'))
fit, stop = internal_split(frames['train'])
assert set(fit.sha256) | set(stop.sha256) == set(frames['train'].sha256)
assert not set(fit.patient) & set(stop.patient)
for name in ['selection', 'calibration', 'test']:
    assert not set(frames[name].patient) & (set(fit.patient) | set(stop.patient))

with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    Image.new('L', (120, 60), 127).save(folder / 'synthetic.png')
    tensor = image_tensor(folder / 'synthetic.png')
    assert tensor.shape == (3, 320, 320) and torch.isfinite(tensor).all()
    # Confirm the standalone loader rejects accidental/corrupt checkpoint changes.
    (folder / 'fake.pt').write_bytes(b'not weights')
    config = dict(image_size=320, aggregation='mean_raw_sigmoid', members=[dict(checkpoint='fake.pt', sha256='0' * 64)])
    (folder / 'inference_config.json').write_text(json.dumps(config))
    try:
        load_config(folder)
    except ValueError:
        pass
    else:
        raise AssertionError('Checkpoint corruption not rejected')

test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_selection.py', '-v'], capture_output=True, text=True)
if test.returncode:
    raise RuntimeError(test.stdout + test.stderr)
for file in HERE.glob('*.py'):
    compile(file.read_text(encoding='utf-8'), str(file), 'exec')
code = (HERE / 'submitted_source.py').read_text(encoding='utf-8')
report = dict(status='passed', scope='CPU synthetic gradient/320-pixel forward, frozen cohort/internal split checks, six development-selection tests, preprocessing and checkpoint integrity rejection; not Kaggle training execution',
              models=records, fit_images=len(fit), early_stop_images=len(stop), fit_groups=int(fit.patient.nunique()),
              early_stop_groups=int(stop.patient.nunique()), source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(),
              seconds=time.monotonic() - started, tests=test.stderr.strip(), torch=torch.__version__)
(HERE / 'local_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
