"""Cross-check compact published records; does not re-evaluate images or refit models."""
import json
import math
from pathlib import Path


def audit(root=Path('.')):
    root=Path(root)
    def read(name):
        return json.loads((root/'results'/name).read_text(encoding='utf-8'))
    def same(a,b,label):
        if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):
            raise ValueError('Inconsistent '+label)
    candidates=[]
    checks=[]
    for size in (224,320):
        prefix=f'benchmark{size}'
        result=read(prefix+'_metrics.json')
        config=read(prefix+'_selection.json')
        trials=read(prefix+'_all_trial_results.json')
        verification=read(prefix+'_cpu_verification.json')
        expected={(a,s) for a in ('efficientnet_b0','densenet121','resnet18') for s in (42,123,2026)}
        if len(trials)!=9 or {(t['architecture'],t['seed']) for t in trials}!=expected:
            raise ValueError('Missing or duplicate trials')
        means={a:sum(t['selection_auroc'] for t in trials if t['architecture']==a)/3
               for a in sorted({t['architecture'] for t in trials})}
        selected=min(means,key=lambda a:(-means[a],a))
        if result['selected_architecture']!=selected or config['architecture']!=selected:
            raise ValueError('Selection rule mismatch')
        for a,mean in means.items():
            same(mean,config['selection_architecture_mean_auroc'][a],'selection mean')
            candidates.append((mean,size,a))
        same(config['threshold'],result['threshold'],'frozen threshold')
        same(config['temperature'],result['temperature'],'frozen temperature')
        metrics=[result['test'],result['raw_test_at_0_5']]
        metrics.extend(t[k] for t in trials for k in ('raw_test','calibrated_test','raw_test_at_0_5'))
        for m in metrics:
            tn,fp,fn,tp=[m[k] for k in ('tn','fp','fn','tp')]
            if any(not isinstance(v,int) or v<0 for v in (tn,fp,fn,tp)) or tn+fp!=231 or fn+tp!=387:
                raise ValueError('Unexpected confusion counts')
            same(m['accuracy'],(tn+tp)/618,'accuracy')
            same(m['sensitivity'],tp/387,'sensitivity')
            same(m['specificity'],tn/231,'specificity')
        if verification['status']!='passed' or verification['images']!=618 or verification['changed_predictions']!=0:
            raise ValueError('CPU verification did not pass')
        if verification['max_absolute_score_difference']>verification['score_tolerance']:
            raise ValueError('CPU discrepancy exceeds tolerance')
        if len(verification['checkpoints'])!=9 or not all(c['strict_load_and_forward'] for c in verification['checkpoints']):
            raise ValueError('Checkpoint verification incomplete')
        stops={t['stop_reason'] for t in trials}
        if stops!={'early_stopping'}:
            raise ValueError('Report early-stopping claim requires revision')
        checks.append(f'{size}: nine trials, development choice, frozen parameters, 29 confusion records and CPU summary agree')
    for size,expected_changes in ((224,1),(320,0)):
        calibration=read(f'benchmark{size}_calibration_reproduction.json')
        boundary=read(f'benchmark{size}_calibration_boundary.json')
        if calibration['images']!=482 or boundary['images']!=482:
            raise ValueError('Calibration cohort count mismatch')
        if calibration['changed_frozen_calibration_predictions']!=expected_changes or len(boundary['changed_images'])!=expected_changes:
            raise ValueError('Calibration classification discrepancy record mismatch')
        if calibration['status']!=boundary['strict_verification_status']:
            raise ValueError('Boundary review must preserve strict verification status')
        if calibration['test_predictions_accessed'] or calibration['frozen_config_changed'] or boundary['frozen_config_changed']:
            raise ValueError('Calibration-only frozen configuration claim violated')
        same(calibration['saved_threshold'],boundary['cutoff'],'boundary cutoff')
        same(calibration['max_frozen_score_difference'],boundary['max_absolute_score_difference'],'boundary score error')
        if boundary['reread_saved_score_decisions_differ_from_stored']!=0:
            raise ValueError('CSV round-trip decision claim requires revision')
        if size==224:
            row=boundary['changed_images'][0]
            same(row['saved_margin'],0,'cutoff equality')
            same(row['cpu_margin'],row['cpu_score']-boundary['cutoff'],'CPU cutoff margin')
            if row['cpu_margin']>=0 or calibration['status']=='passed':
                raise ValueError('Comparator strict failure must remain explicit')
        elif calibration['status']!='passed':
            raise ValueError('Selected model calibration reproduction did not pass')
        checks.append(f'{size}: calibration reproduction and boundary records agree; strict status preserved')
    selection=read('resolution_development_selection.json')
    best=min(candidates,key=lambda x:(-x[0],x[1],x[2]))
    if (selection['selected']['image_size'],selection['selected']['architecture'])!=(best[1],best[2]):
        raise ValueError('Combined development selection mismatch')
    uncertainty=read('resolution_paired_uncertainty.json')
    paired=uncertainty['paired_comparison']
    for key,delta in paired['difference'].items():
        same(delta,paired['followup'][key]-paired['baseline'][key],'paired difference')
        same(paired['followup'][key],read('benchmark320_metrics.json')['test'][key],'320 comparison metric')
        same(paired['baseline'][key],read('benchmark224_metrics.json')['test'][key],'224 comparison metric')
        lo,hi=paired['paired_cluster_bootstrap_95ci'][key]
        if not math.isfinite(lo+hi) or lo>hi or not lo<=0<=hi:
            raise ValueError('Report claim that paired intervals include zero requires revision')
    stability=read('resolution_selection_stability.json')
    if sum(stability['winner_counts'].values())!=stability['usable_replicates']:
        raise ValueError('Bootstrap winner counts do not sum')
    for candidate,count in stability['winner_counts'].items():
        same(count/stability['usable_replicates'],stability['winner_fractions'][candidate],'winner fraction')
    checks.extend(['Combined choice agrees with all six development candidate means',
                   'Paired point differences agree with saved ensemble metrics; all reported paired intervals include zero',
                   'Selection-bootstrap winner counts and fractions agree'])
    return dict(status='passed',checks=checks,
        scope='Internal consistency of compact published records only; not an independent recomputation of image-level AUROC or bootstrap intervals',
        limitations=['Raw per-image prediction archives remain on Kaggle; local records came from visible logs and previews',
                     'CPU verification reloaded images and checkpoints on Kaggle; this audit checks its saved summary',
                     'Original GPU raw calibration probabilities were not exported; fresh CPU scores support a separate refit with one comparator boundary disagreement',
                     'Test sampling intervals omit training and selection uncertainty; filename groups are proxies'])


if __name__=='__main__':
    result=audit(Path(__file__).resolve().parent)
    target=Path(__file__).resolve().parent/'results/research_record_audit.json'
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
