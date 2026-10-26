"""Check captured compact follow-up records; not independent image-level recomputation."""
import hashlib
import json
from pathlib import Path


def audit(folder=None):
    folder=Path(folder) if folder else Path(__file__).resolve().parent
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    choice=read('development_choice.json'); report=read('summary.json'); run=read('run_record.json')
    chosen=choice['selected']; records=choice['candidates']
    if len(records)!=12 or len({r['candidate'] for r in records})!=12: raise ValueError('Candidate list incomplete')
    if chosen['correct']!=max(r['correct'] for r in records): raise ValueError('Incorrect development choice')
    if any(abs(r['oof_accuracy']-r['correct']/r['images'])>1e-12 for r in records): raise ValueError('Accuracy count mismatch')
    if report['selected_candidate']!=chosen['candidate'] or report['members']!=chosen['member_count']: raise ValueError('Model choice changed')
    if report['training_performed'] is not False or choice['test_predictions_accessed'] is not False: raise ValueError('Protocol mismatch')
    cm=report['confusion']; n=sum(cm.values())
    if n!=618: raise ValueError('Test size changed')
    m=report['test_metrics']
    expected=dict(accuracy=(cm['tn']+cm['tp'])/n,sensitivity=cm['tp']/(cm['tp']+cm['fn']),specificity=cm['tn']/(cm['tn']+cm['fp']))
    if any(abs(m[k]-v)>1e-12 for k,v in expected.items()): raise ValueError('Confusion metrics disagree')
    paired=report['paired_comparison']['paired_comparison']
    if paired['images']!=618 or paired['filename_proxy_groups']!=426: raise ValueError('Paired cohort mismatch')
    for key,value in m.items():
        if abs(paired['followup'][key]-value)>1e-12: raise ValueError('Paired follow-up metric mismatch')
        if abs(paired['difference'][key]-(value-paired['baseline'][key]))>1e-12: raise ValueError('Paired difference mismatch')
    cli=report['cli_verification']
    if not cli['passed'] or cli['images']!=2 or len(cli['checks'])!=2: raise ValueError('CLI verification incomplete')
    if any(c['changed_prediction']!=0 or c['score_difference']>1e-4 for c in cli['checks']): raise ValueError('CLI tolerance failed')
    nb=read('pneumonia_accuracy_experiment.ipynb')
    code=''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code').replace('\r\n','\n')
    if hashlib.sha256(code.encode()).hexdigest()!=run['source_sha256_lf']: raise ValueError('Submitted source changed')
    result=dict(status='passed',candidate_count=12,test_images=n,proxy_groups=426,
                source_sha256_lf=run['source_sha256_lf'],
                scope='Compact-record arithmetic and source consistency, not recomputation of image-level AUROC or bootstrap intervals')
    (folder/'result_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__': audit()
