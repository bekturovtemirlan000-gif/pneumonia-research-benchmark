"""Generate the prespecified 320-pixel arm without changing the running 224 arm."""
import argparse
import hashlib
import json
from pathlib import Path


def generate(minutes):
    if not 30 <= minutes <= 180:
        raise ValueError('Additional training budget must be 30 to 180 minutes')
    base=Path('research_benchmark.py').read_text(encoding='utf-8')
    changes={
      '(224,224)':'(320,320)',
      'letterbox 224':'letterbox 320',
      'deadline=started+270*60':f'deadline=started+{minutes}*60',
      "'training_budget_minutes':270":f"'training_budget_minutes':{minutes},'image_size':320",
      "pneumonia_benchmark_outputs":"pneumonia_resolution320_outputs",
      "pneumonia_benchmark_cache":"pneumonia_resolution320_cache",
      "config={'architecture':selected_arch":"config={'image_size':320,'architecture':selected_arch",
    }
    source=base
    for old,new in changes.items():
        if old not in source:
            raise ValueError('Base template changed: '+old)
        source=source.replace(old,new)
    compile(source,'research_resolution320.py','exec')
    Path('research_resolution320.py').write_text(source,encoding='utf-8')
    notebook={'nbformat':4,'nbformat_minor':5,
       'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
       'cells':[{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                 'source':source.splitlines(keepends=True)}]}
    Path('pneumonia_resolution320.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    record=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(source.encode()).hexdigest(),
        training_budget_minutes=minutes,image_size=320,changes=changes)
    Path('artifacts/resolution320_generation.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--minutes',type=int,required=True)
    args=parser.parse_args(); generate(args.minutes)
