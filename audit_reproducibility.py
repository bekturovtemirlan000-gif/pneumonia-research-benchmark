"""Check notebook/source consistency without executing training or reading test scores."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DIRECT = {
    'pneumonia_calibration_boundary.ipynb': 'inspect_calibration_boundary.py',
    'pneumonia_kaggle.ipynb': 'train.py',
    'pneumonia_ensemble_kaggle.ipynb': 'train_ensemble.py',
    'pneumonia_research_benchmark.ipynb': 'research_benchmark.py',
    'pneumonia_resolution320.ipynb': 'research_resolution320.py',
    'pneumonia_external_source_inspection.ipynb': 'external_source_inspection.py',
}
EMBEDDED = [
    'pneumonia_export_verification.ipynb',
    'pneumonia_export_verification_320.ipynb',
    'pneumonia_paired_comparison.ipynb',
    'pneumonia_resolution_analysis.ipynb',
    'pneumonia_calibration_reproduction.ipynb',
]
FROZEN = {
    'pneumonia_calibration_boundary.ipynb': '1d9ae3b59f17ffd61537ebb90408fb1a4855a54ea9a1062fd5906f25f9ee9651',
    'pneumonia_calibration_reproduction.ipynb': '78b068f73f23a7882f12c241512266ef650c232b0787a7247237875a30bb2a0e',
    'pneumonia_research_benchmark.ipynb': '02ac0f88e6c568eac2239bd712f5bbdf3c8a98732ac2a9be59b091c91b95405f',
    'pneumonia_resolution320.ipynb': '0e3d7cf1bdedbfc43957c0cb6cdb0b0763f53b74fa181eb36e3bd24a31af6913',
    'pneumonia_export_verification.ipynb': '04c689993b783902680e82f03bd3ec6b251c3b22afdea17a3989a973119b5d0b',
    'pneumonia_export_verification_320.ipynb': '63c6d4521210a374be97290be3dd41e09fac7ce3f9864b9ca1221997be57c582',
    'pneumonia_resolution_analysis.ipynb': 'd5554d25dd3726fde0eada34dc91f86d0740c5ca445de416fbf1da567ac17609',
    'pneumonia_paired_comparison.ipynb': 'cd4b757e91560bbcedc8833c7cc0e3d57830beef96778129fb2bd40c2702ac33',
    'pneumonia_external_source_inspection.ipynb': '4e6ba61555155bee0be868c4c706fef56b47423320d4c0f9c097de8b41a16b60',
    'pneumonia_similarity_audit.ipynb': '1f3d636e18e2f7c4292b738d8391649ba317c3a45dbb2072c6b07cddb8a49698',
}


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def audit(root=Path('.')):
    root=Path(root)
    records=[]
    for name in [*DIRECT,*EMBEDDED,'pneumonia_similarity_audit.ipynb']:
        notebook=json.loads((root/name).read_text(encoding='utf-8'))
        source=''.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code')
        source=source.replace('\r\n','\n')
        tree=ast.parse(source)
        checks={}
        if name in DIRECT:
            checks[DIRECT[name]]=source==(root/DIRECT[name]).read_text(encoding='utf-8')
        elif name in EMBEDDED:
            assignments=[node for node in tree.body if isinstance(node,ast.Assign)
                         and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='sources']
            if len(assignments)!=1:
                raise ValueError('Expected one embedded sources mapping: '+name)
            for filename,content in ast.literal_eval(assignments[0].value).items():
                checks[filename]=content==(root/filename).read_text(encoding='utf-8')
        else:
            base=(root/'research_benchmark.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
            extra=(root/'similarity_audit.py').read_text(encoding='utf-8').replace('from research_benchmark import split_cohorts, patient_id\n','')
            checks['combined_source_ast']=ast.dump(tree)==ast.dump(ast.parse(base+'\n'+extra))
        checksum=digest(source)
        if name in FROZEN:
            checks['frozen_submitted_source']=checksum==FROZEN[name]
        if not all(checks.values()):
            raise ValueError(f'Source inconsistency in {name}: {checks}')
        records.append(dict(notebook=name,code_sha256_lf=checksum,checks=checks))
    return dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),
                hash_convention='UTF-8 notebook code cells concatenated in order, CRLF normalized to LF',
                scope='Local source consistency; does not independently attest remote execution or results',
                similarity_historical_raw_crlf_sha256='d0dc2c236117a72d183636d9431ec2cf7aa19d416ff06573cdb09e716677915f',
                notebooks=records)


if __name__=='__main__':
    report=audit()
    Path('results').mkdir(exist_ok=True)
    Path('results/source_consistency_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f"Passed source checks for {len(report['notebooks'])} notebooks")
