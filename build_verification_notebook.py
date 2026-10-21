"""Embed the exact local verification and descriptive-analysis sources for Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path


def build(resolution=224):
    if resolution not in (224,320):
        raise ValueError('Only the two planned resolutions are supported')
    names = ['predict.py', 'validate_artifacts.py', 'validate_benchmark.py',
             'verify_exported_model.py', 'analyze_predictions.py', 'inspect_threshold_stability.py']
    sources = {name: Path(name).read_text(encoding='utf-8') for name in names}
    source = '''from pathlib import Path
import sys,json
code=Path('/kaggle/working/verification_code'); code.mkdir(exist_ok=True)
'''
    source += 'sources=' + repr(sources) + '\n'
    source += '''for name,source in sources.items(): (code/name).write_text(source)
sys.path.insert(0,str(code))
from verify_exported_model import verify
from analyze_predictions import analyze
from inspect_threshold_stability import inspect
matches=[p.parent for p in Path('/kaggle/input').rglob('inference_config.json') if p.parent.name=='pneumonia_benchmark_outputs']
if len(matches)!=1: raise ValueError('Attach the completed three-architecture benchmark output')
roots=sorted({p.parent for p in Path('/kaggle/input').rglob('train') if (p/'NORMAL').is_dir() and (p.parent/'test'/'PNEUMONIA').is_dir()},key=lambda p:(len(p.parts),str(p)))
if not roots: raise ValueError('Attach the source chest X-ray dataset')
out=Path('/kaggle/working/verification_outputs'); out.mkdir(exist_ok=True)
analysis=analyze(matches[0],out)
stability=inspect(matches[0])
(out/'threshold_stability.json').write_text(json.dumps(stability,indent=2))
print('THRESHOLD_STABILITY',json.dumps(stability),flush=True)
verify(matches[0],roots[0],out)
'''
    suffix='' if resolution==224 else '_320'
    if resolution==320:
        source=source.replace('pneumonia_benchmark_outputs','pneumonia_resolution320_outputs')
        source=source.replace('completed three-architecture benchmark output','completed 320-pixel benchmark output')
        source=source.replace('verification_outputs','verification320_outputs')
    compile(source,'verification_kaggle_source.py','exec')
    Path(f'artifacts/verification{suffix}_kaggle_source.py').write_text(source,encoding='utf-8')
    notebook={'nbformat':4,'nbformat_minor':5,
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                        'source':source.splitlines(keepends=True)}]}
    Path(f'pneumonia_export_verification{suffix}.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    hashes['combined_source']=hashlib.sha256(source.encode()).hexdigest()
    Path(f'artifacts/verification{suffix}_source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(hashes,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--resolution',type=int,choices=[224,320],default=224)
    build(parser.parse_args().resolution)
