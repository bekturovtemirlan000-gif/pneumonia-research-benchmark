"""Build the exact self-contained Kaggle training source and prospective protocol."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
protocol_path = HERE / 'protocol.json'
if not protocol_path.exists():
    protocol = dict(
        recorded_utc=datetime.now(timezone.utc).isoformat(), local_date='2026-11-15',
        session_started_utc='2026-11-14T18:18:30Z', approximate_work_deadline_utc='2026-11-14T23:18:30Z',
        objective='Highest overall accuracy on development data; nine fresh training runs and honest GitHub results',
        source_dataset='paultimothymooney/chest-xray-pneumonia', source_manifest_version=360888746,
        source_previous_model='https://www.kaggle.com/code/temirlanbekturov/pneumonia-research-resolution-320?scriptVersionId=360888746',
        frozen_counts=dict(train=3815, selection=456, calibration=482, test=618),
        internal_holdout='From original training only: hash-order proxy groups within class with salt 20261115, reserve every fifth group. Same split for every trial.',
        recipes={
            'b0_standard': dict(architecture='efficientnet_b0', loss='Unweighted BCE', mixup_alpha=0, label_smoothing=0, color_jitter=0, weight_decay=.0001),
            'b0_regularized': dict(architecture='efficientnet_b0', loss='Unweighted BCE with soft targets', mixup_alpha=.2, label_smoothing=.05, color_jitter=.15, weight_decay=.0005),
            'v2s_regularized': dict(architecture='efficientnet_v2_s', loss='Unweighted BCE with soft targets', mixup_alpha=.2, label_smoothing=.05, color_jitter=.15, weight_decay=.0005)},
        seeds=[42, 123, 2026], image_size=320, batch_size=24, epochs_max=36, minimum_epochs=8, patience=8,
        pretrained='Official torchvision ImageNet1K_V1 for each architecture; all layers fine-tuned from start',
        optimizer='AdamW backbone LR 1e-4, head LR 1e-3; cosine schedule 36 epochs to 1e-6; norm clipping 5; float16 AMP training and float32 evaluation',
        augmentation='All: affine rotation +/-7 degrees, translate +/-3%, scale 0.95-1.05, bilinear sampling. Regularized: brightness and contrast +/-15%, mixup alpha 0.2, label smoothing 0.05.',
        preprocessing='RGB; full-image bicubic letterbox to 320; ImageNet channel normalization. No test-time augmentation or cropping.',
        checkpoint_selection='Maximum internal-holdout correct count at cutoff 0.5; tie lower log loss; only complete epochs qualify. Early stop after eight stale epochs and at least eight complete epochs unless time budget intervenes.',
        candidates='Four fixed equal-weight raw-sigmoid ensembles: each recipe across three seeds, and all nine members',
        candidate_selection='Five StratifiedGroupKFold folds seed 20261115 on original 456 selection images. Fit max-accuracy cutoff on four folds, score fifth; maximize pooled correct count. Tie fewer members, then candidate name.',
        threshold_tie_rule='Lowest cutoff among 0, unique-score midpoints and nextafter(max_score,+inf)',
        final_threshold='Fit maximum accuracy cutoff on separate 482 calibration images; no temperature scaling; freeze configuration and checkpoint hashes before test inference.',
        evaluation='Only selected ensemble scored on previously inspected 618-image source test. Also report fixed 0.5 cutoff descriptively; it cannot replace the frozen cutoff based on outcomes.',
        comparison='Original selected 320 EfficientNet-B0 model, paired 1000 seed-42 filename-group bootstrap; preserve original model and all historical results.',
        training_budget_minutes=225,
        allocation='225 minutes from after image audit divided across remaining trials; rotated recipe order across seeds; unused early-stop time redistributed. Queue, audit and final evaluation excluded; total work target about five hours.',
        incomplete_policy='Save partial trials and fail if all nine lack a complete trained checkpoint; record time-limited runs without claiming full convergence.',
        verification='All decoded pixels and frozen cohort digests; local meaningful selection tests and architecture forward checks; two actual exported CPU CLI subprocesses; separate full test-cohort CPU reproduction after run when time allows.',
        limitations=['All source cohorts previously inspected: exploratory, not confirmatory or external validation',
                    'New checkpoint holdout prevents this run from selecting checkpoints on selection cohort; it does not erase prior experimentation on these cohorts',
                    'Candidate selection optimism and training uncertainty are not captured by conditional test bootstrap',
                    'Internal holdout reduces optimization training size; this experiment changes multiple factors relative to original model and is not a controlled attribution of all differences',
                    'Filename groups are patient proxies, similarity audit not fully adjudicated',
                    'Accuracy depends on class prevalence and may reduce sensitivity',
                    'No clinical safety or perfect-accuracy claim; report failures and negative results'],
        references=['https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_v2_s.html',
                    'https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html',
                    'https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html',
                    'https://scikit-learn.org/stable/modules/cross_validation.html'])
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
old_config = json.loads((ROOT / 'results/benchmark320_selection.json').read_text(encoding='utf-8'))
old_verification = json.loads((ROOT / 'results/benchmark320_cpu_verification.json').read_text(encoding='utf-8'))
previous_lock = dict(config={k: old_config[k] for k in ['architecture', 'selected_checkpoints', 'temperature', 'threshold']},
                     checkpoints={r['trial']: r['sha256'] for r in old_verification['checkpoints']})
(HERE / 'previous_lock.json').write_text(json.dumps(previous_lock, indent=2), encoding='utf-8')
sources = {name: (HERE / name).read_text(encoding='utf-8') for name in ['selection.py', 'model.py', 'train_experiment.py']}
sources.update({name: (ROOT / name).read_text(encoding='utf-8') for name in ['research_benchmark.py', 'compare_experiments.py', 'analyze_benchmark_comparison.py']})
code = "from pathlib import Path\nimport sys\ncode_dir=Path('/kaggle/working/november_training_code');code_dir.mkdir(exist_ok=True)\n"
code += 'sources=' + repr(sources) + '\nprotocol=' + repr(protocol) + '\nprevious_lock=' + repr(previous_lock) + '\n'
code += """for name, source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from train_experiment import run
root=Path('/kaggle/input')
previous=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
data=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(previous)!=1 or not data: raise ValueError('Attach original 320 benchmark outputs version 360888746 and chest-X-ray source dataset')
run(data[0],previous[0],'/kaggle/working/pneumonia_november_outputs',protocol,previous_lock)
"""
compile(code, 'submitted_source.py', 'exec')
notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True))])
(HERE / 'pneumonia_training.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
(HERE / 'submitted_source.py').write_text(code, encoding='utf-8')
print(json.dumps(dict(source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(), characters=len(code))))
