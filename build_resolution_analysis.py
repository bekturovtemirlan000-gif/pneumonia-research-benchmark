"""Package the planned development-only selection and frozen resolution comparison."""
import hashlib
import json
from pathlib import Path


def build():
    names=['select_resolution.py','inspect_selection_stability.py',
           'compare_experiments.py','analyze_benchmark_comparison.py']
    sources={name:Path(name).read_text(encoding='utf-8') for name in names}
    code="""from pathlib import Path
import json,sys
destination=Path('/kaggle/working/resolution_analysis_code');destination.mkdir(exist_ok=True)
"""
    code+='sources='+repr(sources)+'\n'
    code+="""for name,text in sources.items(): (destination/name).write_text(text)
sys.path.insert(0,str(destination))
from select_resolution import select
from inspect_selection_stability import inspect
from analyze_benchmark_comparison import analyze
root=Path('/kaggle/input')
def find_output(name):
    matches=[p.parent for p in root.rglob('inference_config.json') if p.parent.name==name]
    if len(matches)!=1: raise ValueError('Attach exactly one completed output: '+name)
    return matches[0]
low=find_output('pneumonia_benchmark_outputs')
high=find_output('pneumonia_resolution320_outputs')
out=Path('/kaggle/working/resolution_comparison_outputs');out.mkdir(exist_ok=True)
# Save the development-only decision before reading test predictions.
decision=select(low,high)
(out/'development_selection.json').write_text(json.dumps(decision,indent=2))
print('DEVELOPMENT_SELECTION',json.dumps(decision),flush=True)
stability=inspect(low,high)
(out/'selection_stability.json').write_text(json.dumps(stability,indent=2))
print('SELECTION_STABILITY',json.dumps(stability),flush=True)
report=analyze(low,high,out)
report['experiment_mapping']={'baseline':'224-pixel development-selected ensemble','followup':'320-pixel development-selected ensemble'}
(out/'comparison.json').write_text(json.dumps(report,indent=2))
print('RESOLUTION_COMPARISON_DONE',json.dumps(report),flush=True)
"""
    compile(code,'resolution_analysis_kaggle_source.py','exec')
    notebook={'nbformat':4,'nbformat_minor':5,
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                        'source':code.splitlines(keepends=True)}]}
    Path('pneumonia_resolution_analysis.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/resolution_analysis_kaggle_source.py').write_text(code,encoding='utf-8')
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
    hashes['combined_source']=hashlib.sha256(code.encode()).hexdigest()
    Path('artifacts/resolution_analysis_source_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(hashes,indent=2))


if __name__=='__main__':
    build()
