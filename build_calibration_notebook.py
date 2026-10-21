"""Build the calibration-only Kaggle notebook from reviewed local source modules."""
import hashlib
import json
from pathlib import Path

NAMES = ['predict.py', 'validate_artifacts.py', 'validate_benchmark.py', 'verify_exported_model.py', 'verify_calibration_fit.py']
PREAMBLE = "from pathlib import Path\nimport json,sys\ndest=Path('/kaggle/working/calibration_code');dest.mkdir(exist_ok=True)\n"
RUNNER = "for name,source in sources.items(): (dest/name).write_text(source)\nsys.path.insert(0,str(dest))\nfrom verify_calibration_fit import verify\nroot=Path('/kaggle/input')\nroots=sorted({p.parent for p in root.rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))\nif not roots: raise ValueError('Attach original chest X-ray data')\nreports=[]\nfor size,name in [(224,'pneumonia_benchmark_outputs'),(320,'pneumonia_resolution320_outputs')]:\n    matches=[p.parent for p in root.rglob('inference_config.json') if p.parent.name==name]\n    if len(matches)!=1: raise ValueError('Attach exactly one completed output: '+name)\n    reports.append(verify(matches[0],roots[0],Path('/kaggle/working/calibration_reproduction_outputs')/str(size)))\nprint('CALIBRATION_ALL_DONE',json.dumps(reports),flush=True)\nif any(r['status']!='passed' for r in reports): raise RuntimeError('Review numerical discrepancies; frozen configs unchanged')\n"

def build():
    root=Path(__file__).resolve().parent
    sources={name:(root/name).read_text(encoding='utf-8') for name in NAMES}
    source=PREAMBLE+'sources='+repr(sources)+'\n'+RUNNER
    compile(source,'calibration_reproduction.py','exec')
    notebook={'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':source.splitlines(keepends=True)}]}
    (root/'pneumonia_calibration_reproduction.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    (root/'artifacts').mkdir(exist_ok=True)
    (root/'artifacts/calibration_reproduction_source.py').write_text(source,encoding='utf-8')
    print(hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__': build()
