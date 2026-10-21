"""Package the frozen selected-model inference CLI smoke test for Kaggle CPU."""
import hashlib
import json
from pathlib import Path


def build():
    sources = {name: Path(name).read_text(encoding='utf-8') for name in ['predict.py', 'verify_inference_cli.py']}
    lock = json.loads(Path('results/external_model_lock.json').read_text(encoding='utf-8'))
    code = "from pathlib import Path\nimport sys\ndestination=Path('/kaggle/working/inference_cli_code');destination.mkdir(exist_ok=True)\n"
    code += 'sources=' + repr(sources) + '\nlock=' + repr(lock) + '\n'
    code += """for name,source in sources.items(): (destination/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(destination))
from verify_inference_cli import verify
root=Path('/kaggle/input')
models=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
datasets=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if len(models)!=1 or not datasets:
    raise ValueError('Attach one 320-pixel training output and one original image dataset')
verify(models[0],datasets[0],destination,lock,'/kaggle/working/inference_cli_outputs')
"""
    compile(code, 'inference_cli_source.py', 'exec')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                    cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[], source=code.splitlines(keepends=True))])
    Path('pneumonia_inference_cli_smoke.ipynb').write_text(json.dumps(notebook, indent=2), encoding='utf-8')
    Path('artifacts/inference_cli_source.py').write_text(code, encoding='utf-8')
    print(hashlib.sha256(code.encode()).hexdigest())


if __name__ == '__main__':
    build()
