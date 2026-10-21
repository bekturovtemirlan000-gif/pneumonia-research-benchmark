"""Package frozen geometry export verification and paired source-test uncertainty."""
import hashlib
import json
from pathlib import Path


def build():
    sources={name:Path(name).read_text(encoding='utf-8') for name in
        ['compare_experiments.py','analyze_benchmark_comparison.py','analyze_geometry_control.py']}
    code="from pathlib import Path\nimport sys\ndestination=Path('/kaggle/working/geometry_analysis_code');destination.mkdir(exist_ok=True)\n"
    code+='sources='+repr(sources)+'\n'
    code+="""for name,source in sources.items(): (destination/name).write_text(source,encoding='utf-8')
sys.path.insert(0,str(destination))
from analyze_geometry_control import run
root=Path('/kaggle/input')
controls=[p.parent for p in root.rglob('geometry_model.json')]
models=[p.parent for p in root.rglob('inference_config.json') if p.parent.name=='pneumonia_resolution320_outputs']
if len(controls)!=1 or len(models)!=1:
    raise ValueError('Attach one completed geometry-control output and one completed 320-pixel training output')
run(controls[0],models[0],'/kaggle/working/geometry_analysis_outputs')
"""
    compile(code,'geometry_analysis_source.py','exec')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        cells=[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(keepends=True))])
    Path('pneumonia_geometry_analysis.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    Path('artifacts/geometry_analysis_source.py').write_text(code,encoding='utf-8')
    print(hashlib.sha256(code.encode()).hexdigest())


if __name__=='__main__':
    build()
