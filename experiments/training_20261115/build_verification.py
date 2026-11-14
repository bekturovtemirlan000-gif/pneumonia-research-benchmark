"""Build CPU reproduction only after a concrete training configuration is frozen."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(config_sha256):
    if len(config_sha256) != 64 or any(c not in '0123456789abcdef' for c in config_sha256):
        raise ValueError('Expected frozen configuration SHA256')
    sources = {name: (HERE / name).read_text(encoding='utf-8') for name in ['model.py', 'selection.py', 'verify_model.py', 'export_predictions.py']}
    sources['research_benchmark.py'] = (ROOT / 'research_benchmark.py').read_text(encoding='utf-8')
    training_hash = hashlib.sha256((HERE / 'submitted_source.py').read_text(encoding='utf-8').encode()).hexdigest()
    code = "from pathlib import Path\nimport hashlib, sys\ncode_dir=Path('/kaggle/working/november_verify_code');code_dir.mkdir(exist_ok=True)\n"
    code += 'sources=' + repr(sources) + '\n'
    code += 'expected_config_sha256=' + repr(config_sha256) + '\n'
    code += 'training_source_sha256=' + repr(training_hash) + '\n'
    code += """for name,source in sources.items(): (code_dir/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(code_dir))
from verify_model import verify
root=Path('/kaggle/input')
models=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_november_outputs']
data=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(models)!=1 or not data: raise ValueError('Attach completed November training output and original chest-X-ray dataset')
if hashlib.sha256((models[0]/'inference_config.json').read_bytes()).hexdigest()!=expected_config_sha256:
    raise ValueError('Attached output does not match the observed frozen training configuration')
for name in ['model.py','selection.py']:
    if (models[0]/name).read_text(encoding='utf-8')!=sources[name]:
        raise ValueError('Exported inference or selection source differs from independently packaged source')
verify(models[0],data[0],'/kaggle/working/november_cpu_verification',training_source_sha256)
"""
    compile(code, 'submitted_verification.py', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True))])
    (HERE / 'pneumonia_cpu_verification.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    (HERE / 'submitted_verification.py').write_text(code, encoding='utf-8')
    print(json.dumps(dict(verification_source_sha256_lf=hashlib.sha256(code.encode()).hexdigest(), frozen_config_sha256=config_sha256)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-sha256', required=True)
    build(parser.parse_args().config_sha256)
