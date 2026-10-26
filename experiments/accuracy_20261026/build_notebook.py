import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
protocol_path=HERE/'protocol.json'
if not protocol_path.exists():
    protocol=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        objective='Highest overall accuracy on development data, explicitly requested by user',
        method='Reuse original frozen checkpoints; no further training in this stage',
        candidates='12 fixed equal-weight raw-sigmoid combinations: six size/architecture triples, two size-wide nine-member ensembles, three architecture-wide six-member ensembles, and all eighteen',
        selection='Five-fold StratifiedGroupKFold seed 20261026 on original selection partition; fit maximum-accuracy threshold on four folds and score the fifth; rank pooled correct counts',
        selection_ties='Prefer original 320 EfficientNet-B0 candidate, then fewer members, then name',
        final_threshold='Fit maximum accuracy using fresh original-checkpoint inference on separate calibration partition; choose lowest midpoint threshold on ties',
        score_calibration='None; raw averaged sigmoid score is not asserted to be a calibrated probability',
        test='Freeze exported weights and config before test scoring; report all changes including sensitivity losses; no subsequent tuning on observed results',
        comparison='Original selected 320 model; paired 1000 seed-42 filename-group bootstrap on identical test images',
        verification='Original eighteen checkpoint hashes, manifest identity, decoded calibration/test pixels, two hash-selected actual CPU CLI subprocesses at tolerance 1e-4 and zero changed predictions',
        source_versions={'224':360875315,'320':360888746},
        limitations=['Previously inspected selection, calibration and test cohorts: exploratory, not confirmatory',
                    'Original checkpoint selection used all selection images; threshold cross-fitting does not remove that optimism',
                    'Candidate selection optimism and training uncertainty are not covered by test bootstrap intervals',
                    'Filename groups are proxies, not verified patient identities',
                    'Accuracy is prevalence-dependent and may trade away pneumonia sensitivity',
                    'No external or clinical validation; original model remains preserved'],
        reference='https://scikit-learn.org/stable/modules/cross_validation.html')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
protocol=json.loads(protocol_path.read_text())
locks={}
for size in [224,320]:
    report=json.loads((ROOT/f'results/benchmark{size}_cpu_verification.json').read_text())
    for item in report['checkpoints']: locks[f"{size}/{item['trial']}"]=item['sha256']
if len(locks)!=18: raise ValueError('Expected 18 original checkpoint locks')
sources={name:(HERE/name).read_text(encoding='utf-8') for name in ['accuracy_selection.py','predict_accuracy.py','run_accuracy_experiment.py']}
sources.update({name:(ROOT/name).read_text(encoding='utf-8') for name in ['compare_experiments.py','analyze_benchmark_comparison.py']})
code="from pathlib import Path\nimport sys\ncode_dir=Path('/kaggle/working/accuracy_code');code_dir.mkdir(exist_ok=True)\n"
code+='sources='+repr(sources)+'\nprotocol='+repr(protocol)+'\nlocks='+repr(locks)+'\n'
code+="""for name,source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from run_accuracy_experiment import run
root=Path('/kaggle/input')
low=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_benchmark_outputs']
high=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
data=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(low)!=1 or len(high)!=1 or not data: raise ValueError('Attach original 224 and 320 outputs plus original chest-X-ray dataset')
run(low[0],high[0],data[0],'/kaggle/working/pneumonia_accuracy_outputs',protocol,locks)
"""
compile(code,'accuracy_source.py','exec')
notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(HERE/'pneumonia_accuracy_experiment.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
(HERE/'submitted_source.py').write_text(code,encoding='utf-8')
print(json.dumps(dict(source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(),checkpoint_locks=len(locks),notebook=str(HERE/'pneumonia_accuracy_experiment.ipynb'))))
