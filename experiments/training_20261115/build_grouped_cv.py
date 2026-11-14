"""Prespecify and package the secondary grouped training-cohort experiment."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
path = HERE / 'cv_protocol.json'
if not path.exists():
    protocol = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
                    purpose='Secondary internal generalization comparison of the three prespecified training recipes; no change to the primary model selection',
                    prospective_status='Recorded while primary run 369624654 is training, before its candidate selection and test outcomes',
                    cohort='Only the frozen original 3815 training images / 2153 filename proxy groups; no original selection, calibration or test predictions',
                    outer_split='Three-fold StratifiedGroupKFold seed 20261115, same folds across recipes',
                    inner_split='Within each outer training set, every fifth SHA256-ordered proxy group within class using salt 20261115 is held out for checkpoint and cutoff fitting',
                    recipes=['b0_standard','b0_regularized','v2s_regularized'], initialization_seed=42,
                    recipe_definitions='Identical to primary training experiment. Unweighted BCE; optional mixup 0.2, smoothing 0.05, color jitter 0.15; fixed optimizer and preprocessing.',
                    checkpoint='Inner holdout correct count at 0.5, tie lower log loss; no outer image participates',
                    cutoff='Maximum inner-holdout accuracy; freeze checkpoint SHA and cutoff before any outer-fold inference',
                    evaluation='Pool the three out-of-fold predictions per recipe with fold-specific frozen cutoffs. Report pooled accuracy, sensitivity, specificity, Brier, AUROC/AP and per-fold accuracy; pooled ranking scores combine differently scaled models.',
                    inference='This tests individual fitted models with one seed per fold, not the primary three-seed ensemble. No model promotion or threshold change follows these results.',
                    training_budget_minutes=165, epochs_max=36, minimum_epochs=8, patience=8,
                    allocation='Rotate recipe order across folds; divide remaining budget among remaining trials; preserve partial outputs and fail if nine trained checkpoints cannot be completed',
                    uncertainty='No fold-based standard error or significance test: overlapping training sets make folds dependent. No external-generalization confidence claim.',
                    limitations=['Filename group identifiers are unverified patient proxies',
                                 'Original training cohort already used during previous research; this is secondary exploratory analysis',
                                 'A single initialization per fold does not quantify seed uncertainty',
                                 'Checkpoint and cutoff use the same inner holdout; the outer fold excludes those groups',
                                 'The primary candidate and its prespecified evaluation are unchanged'])
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
protocol = json.loads(path.read_text(encoding='utf-8'))
previous_lock = json.loads((HERE / 'previous_lock.json').read_text(encoding='utf-8'))
sources = {name: (HERE / name).read_text(encoding='utf-8') for name in ['model.py','selection.py','train_experiment.py','run_grouped_cv.py']}
sources.update({name: (ROOT / name).read_text(encoding='utf-8') for name in ['research_benchmark.py','compare_experiments.py','analyze_benchmark_comparison.py']})
code = "from pathlib import Path\nimport sys\ncode_dir=Path('/kaggle/working/november_cv_code');code_dir.mkdir(exist_ok=True)\n"
code += 'sources=' + repr(sources) + '\nprotocol=' + repr(protocol) + '\nprevious_lock=' + repr(previous_lock) + '\n'
code += """for name,source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from run_grouped_cv import run
root=Path('/kaggle/input')
previous=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
data=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(previous)!=1 or not data: raise ValueError('Attach original 320 benchmark outputs and original chest-X-ray dataset')
run(data[0],previous[0],'/kaggle/working/pneumonia_november_cv_outputs',protocol,previous_lock)
"""
compile(code, 'submitted_grouped_cv.py', 'exec')
nb = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}},
          cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
(HERE / 'pneumonia_grouped_cv.ipynb').write_text(json.dumps(nb,indent=2),encoding='utf-8')
(HERE / 'submitted_grouped_cv.py').write_text(code,encoding='utf-8')
print(json.dumps(dict(source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(),characters=len(code))))
