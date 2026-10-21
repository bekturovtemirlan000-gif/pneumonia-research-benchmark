"""Cross-check compact published records; does not re-evaluate images or refit models."""
import json
import hashlib
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
    for size in (224,320):
        diagnostic=read(f'benchmark{size}_prediction_diagnostics.json')
        for split,n,positive in (('selection',456,319),('calibration',482,349),('test',618,387)):
            item=diagnostic[split];bins=item['calibration_bins']
            if item['n']!=n or sum(row[2] for row in bins)!=n:
                raise ValueError('Diagnostic bin population mismatch')
            if any(not (0<=a<=1 and 0<=b<=1 and count>0 and count==int(count)) for a,b,count in bins):
                raise ValueError('Invalid reliability bin')
            same(sum(b*count for a,b,count in bins),positive,'bin positive total')
            same(sum(count*abs(a-b) for a,b,count in bins)/n,item['ece_10_equal_width_bins'],'bin ECE')
            same(sum(a*count for a,b,count in bins)/n,((n-positive)*item['class_0_mean_score']+positive*item['class_1_mean_score'])/n,'bin mean score')
        for metric in ('auroc','average_precision','brier_score'):
            same(diagnostic['test'][metric],read(f'benchmark{size}_metrics.json')['test'][metric],'diagnostic test metric')
        if 'diagnostic_log_loss' not in diagnostic['metric_definitions']:
            raise ValueError('Distinct log-loss convention must be disclosed')
        checks.append(f'{size}: reliability-bin counts, positive totals, ECE and mean scores reconcile; log-loss convention labeled')
    lock=read('external_model_lock.json')
    if lock['configuration']!=read('benchmark320_selection.json'):
        raise ValueError('External model lock differs from selected configuration')
    if lock['checkpoints']!=read('benchmark320_calibration_reproduction.json')['checkpoints']:
        raise ValueError('External model checkpoint lock mismatch')
    test_hashes={row['trial']:row['sha256'] for row in read('benchmark320_cpu_verification.json')['checkpoints']}
    if any(test_hashes[row['checkpoint'].split('/')[0]]!=row['sha256'] for row in lock['checkpoints']):
        raise ValueError('External lock disagrees with test checkpoint verification')
    if hashlib.sha256((root/'predict.py').read_text(encoding='utf-8').encode()).hexdigest()!=lock['inference_source_sha256_lf']:
        raise ValueError('External inference source lock mismatch')
    checks.append('Prospective external model lock matches selected config, two checkpoint checks and inference source; no external evaluation implied')
    errors=read('frozen_error_analysis.json')
    overlap=errors['overlap']
    if sum(overlap[k] for k in ('both_correct','both_incorrect','only_224_correct','only_320_correct'))!=errors['images']:
        raise ValueError('Error overlap counts do not cover cohort')
    if overlap['only_224_correct']+overlap['only_320_correct']!=overlap['different_binary_predictions']:
        raise ValueError('Binary disagreement counts do not agree')
    for size in (224,320):
        total=errors['models'][str(size)]
        for metric in ('sensitivity','specificity','brier_score'):
            same(total[metric],read(f'benchmark{size}_metrics.json')['test'][metric],'error analysis metric')
        parts=list(errors['subgroups'][str(size)].values())
        for key in ('images','tn','fp','fn','tp'):
            if sum(part[key] for part in parts)!=total[key]:
                raise ValueError('Subgroup confusion counts do not sum')
        same(sum(part['images']*part['brier_score'] for part in parts)/total['images'],total['brier_score'],'subgroup Brier decomposition')
    for key in ('both_correct','both_incorrect','only_224_correct','only_320_correct'):
        if sum(part[key] for part in errors['overlap_by_source_label'].values())!=overlap[key]:
            raise ValueError('Class-specific overlap does not sum')
    checks.append('Frozen error overlap, class-specific counts and subgroup metric decomposition agree')
    geometry=read('geometry_control.json')
    analysis=read('geometry_analysis.json')
    verification=analysis['export_verification']
    if verification['status']!='passed' or verification['images']!=5371 or verification['changed_classifications']!=0 or verification['max_absolute_score_difference']>verification['score_tolerance']:
        raise ValueError('Geometry coefficient reconstruction failed')
    if geometry['configuration']['fit_partition']!='train' or geometry['configuration']['threshold']!=.5 or geometry['configuration']['hyperparameter_search']:
        raise ValueError('Geometry fitting protocol changed')
    comparison=analysis['comparison']['paired_comparison']
    if comparison['images']!=618 or comparison['filename_proxy_groups']!=426:
        raise ValueError('Geometry paired cohort differs')
    for key,delta in comparison['difference'].items():
        same(comparison['baseline'][key],read('benchmark320_metrics.json')['test'][key],'geometry neural metric')
        same(comparison['followup'][key],geometry['metrics']['test'][key],'geometry control metric')
        same(delta,comparison['followup'][key]-comparison['baseline'][key],'geometry-minus-neural difference')
        lo,hi=comparison['paired_cluster_bootstrap_95ci'][key]
        if not math.isfinite(lo+hi) or lo>hi:
            raise ValueError('Invalid geometry paired interval')
    checks.append('Geometry coefficient verification passed; same 618-image cohort and geometry-minus-neural differences reconcile')
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
