"""Package a read-only artifact export for independent paired-result auditing."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build():
    sources = {name: (HERE/name).read_text(encoding='utf-8') for name in ['export_comparator.py', 'export_predictions.py']}
    sources['compare_experiments.py'] = (ROOT/'compare_experiments.py').read_text(encoding='utf-8')
    lock = json.loads((HERE/'previous_lock.json').read_text(encoding='utf-8'))
    expected = json.loads((HERE/'summary.json').read_text(encoding='utf-8'))['uncertainty']['paired_comparison']['baseline']
    code = "from pathlib import Path\nimport sys\ncode_dir=Path('/kaggle/working/comparator_export_code');code_dir.mkdir(exist_ok=True)\n"
    code += 'sources=' + repr(sources) + '\nlock=' + repr(lock) + '\nexpected_metrics=' + repr(expected) + '\n'
    code += """for name, source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from export_comparator import run
models=[p.parent for p in Path('/kaggle/input').rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
if len(models)!=1: raise ValueError('Attach original completed 320-pixel benchmark version 360888746')
run(models[0],'/kaggle/working/original_comparator_export',lock,expected_metrics)
"""
    compile(code, 'submitted_comparator_export.py', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}},
                    cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    (HERE/'submitted_comparator_export.py').write_text(code,encoding='utf-8')
    (HERE/'pneumonia_comparator_export.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    print(json.dumps(dict(source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(),characters=len(code))))


if __name__ == '__main__': build()
