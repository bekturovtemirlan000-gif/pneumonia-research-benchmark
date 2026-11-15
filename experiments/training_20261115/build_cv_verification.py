"""Package independent CPU checks for an observed completed grouped-CV run."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(summary_sha256):
    if len(summary_sha256) != 64 or any(c not in '0123456789abcdef' for c in summary_sha256):
        raise ValueError('Expected completed CV summary SHA256')
    sources = {name: (HERE / name).read_text(encoding='utf-8') for name in
               ['model.py', 'selection.py', 'verify_model.py', 'verify_grouped_cv.py', 'cv_geometry_control.py', 'export_predictions.py']}
    for name in ['research_benchmark.py', 'compare_experiments.py']:
        sources[name] = (ROOT / name).read_text(encoding='utf-8')
    code = "from pathlib import Path\nimport hashlib, sys\ncode_dir=Path('/kaggle/working/november_cv_verify_code');code_dir.mkdir(exist_ok=True)\n"
    code += 'sources=' + repr(sources) + '\n'
    code += 'expected_summary_sha256=' + repr(summary_sha256) + '\n'
    code += """for name,source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from verify_grouped_cv import verify
root=Path('/kaggle/input')
models=[p.parent for p in root.rglob('summary.json') if p.parent.name=='pneumonia_november_cv_outputs']
data=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(models)!=1 or not data: raise ValueError('Attach completed November grouped-CV output and original chest-X-ray dataset')
for name in ['model.py','selection.py']:
    if (models[0]/name).read_text(encoding='utf-8')!=sources[name]:
        raise ValueError('Exported CV source differs from independently packaged source')
verify(models[0],data[0],'/kaggle/working/november_cv_cpu_verification',expected_summary_sha256)
"""
    compile(code, 'submitted_cv_verification.py', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True))])
    (HERE / 'pneumonia_cv_cpu_verification.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    (HERE / 'submitted_cv_verification.py').write_text(code, encoding='utf-8')
    print(json.dumps(dict(verification_source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(), expected_cv_summary_sha256=summary_sha256)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-sha256', required=True)
    build(parser.parse_args().summary_sha256)
