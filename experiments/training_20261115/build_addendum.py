"""Evidence-based research addendum; requires completed training records."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name): return json.loads((HERE / name).read_text(encoding='utf-8'))


def build():
    result, record, audit, choice = map(read, ['summary.json', 'run_record.json', 'data_audit.json', 'development_choice.json'])
    if record['status'] != 'completed' or result['status'] != 'completed' or len(result['trials']) != 9:
        raise ValueError('Verified completed Kaggle run required before report generation')
    verification = read('cpu_verification.json') if (HERE / 'cpu_verification.json').exists() else None
    cmp = result['uncertainty']['paired_comparison']
    ci = result['uncertainty']['benchmark_cluster_bootstrap_95ci']
    m = result['test']
    output = ROOT / 'output/pdf/pneumonia_november_training_addendum.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles['Normal'].fontName = 'Times-Roman'; styles['Normal'].fontSize = 10.5
    styles['Normal'].leading = 13.5; styles['Normal'].spaceAfter = 7
    for name in ['Title', 'Heading1', 'Heading2']:
        styles[name].fontName = 'Times-Bold'; styles[name].textColor = colors.HexColor('#173b4b')
        styles[name].keepWithNext = True
    styles['Title'].fontSize = 22; styles['Title'].leading = 25
    styles['Heading1'].fontSize = 14; styles['Heading1'].spaceBefore = 10
    styles.add(ParagraphStyle(name='SmallNote', fontName='Times-Roman', fontSize=8.5, leading=11, spaceAfter=5))
    styles.add(ParagraphStyle(name='Cell', fontName='Times-Roman', fontSize=8.5, leading=11))
    story = []
    def p(text, style='Normal'): story.append(Paragraph(text, styles[style]))
    def h(text): p(text, 'Heading1')
    def table(rows, widths):
        content = [[Paragraph(escape(str(value)), styles['Cell']) for value in row] for row in rows]
        t = Table(content, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9f0f3')),
                              ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#8dabb6')),
                              ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d5dfe3')),
                              ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                              ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                              ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.extend([t, Spacer(1, 8)])
    pct = lambda x: f'{100*x:.2f}%'
    p('Development-accuracy selection after regularized deep-learning training', 'Title')
    p('Pediatric pneumonia image classification | Research addendum | 15 November 2026', 'SmallNote')
    p('Exploratory computational study. This document supplements the original October benchmark; it is not a peer-reviewed publication or evidence of clinical diagnostic safety.', 'SmallNote')
    h('Abstract')
    p('<b>Purpose.</b> To evaluate a new development-accuracy-oriented training protocol after an earlier frozen-checkpoint experiment improved development accuracy but reduced source-test accuracy. A perfect classifier was not assumed or promised.')
    p('<b>Methods.</b> Nine fresh transfer-learning trials compared standard EfficientNet-B0, regularized EfficientNet-B0, and regularized EfficientNet-V2-S, each at three seeds. An internal holdout selected checkpoints; four fixed ensemble candidates were ranked by grouped cross-fitted cutoff accuracy on a separate development cohort. A calibration-only cutoff and all model hashes were frozen before source-test inference. All source cohorts had been used in earlier experiments.')
    p(f"<b>Results.</b> Development selection chose {escape(result['selected_candidate'])}: {choice['selected']['correct']}/456 correct ({pct(choice['selected']['cross_fitted_accuracy'])}) under the cutoff cross-fitting procedure. On the 618-image source test, accuracy was {pct(m['accuracy'])} (95% filename-group bootstrap interval {pct(ci['accuracy'][0])} to {pct(ci['accuracy'][1])}), sensitivity {pct(m['sensitivity'])}, specificity {pct(m['specificity'])}, and AUROC {m['auroc']:.4f}. The paired accuracy difference from the original model was {100*cmp['difference']['accuracy']:+.2f} percentage points (95% interval {100*cmp['paired_cluster_bootstrap_95ci']['accuracy'][0]:+.2f} to {100*cmp['paired_cluster_bootstrap_95ci']['accuracy'][1]:+.2f}).")
    p('<b>Interpretation.</b> Fresh accuracy-oriented training did not improve source-test accuracy under the frozen development-selected rule. Brier score improved while discrimination and specificity decreased. Repeated development use, filename-based patient proxies, and absent external testing limit generalization. Conditional bootstrap intervals omit training and model-selection uncertainty.')
    h('Design and data')
    p('This is a retrospective secondary analysis of the existing public research-image resource. The source literature describes pediatric anterior-posterior radiographs from Guangzhou; this project inherits directory labels and does not independently adjudicate clinical diagnosis [1]. The previous decoded-pixel audit retained 5,371 of 5,856 source images. The current run verifies every retained image against the frozen pixel identity and reconstructs the original partitions. No new exclusions are made after observing outcomes.')
    rows = [['Role', 'Images', 'Proxy groups', 'Normal / pneumonia']]
    for name, label in [('fit', 'Parameter fitting'), ('early_stop', 'Checkpoint selection'), ('selection', 'Ensemble selection'), ('calibration', 'Final cutoff fitting'), ('test', 'Exploratory source test')]:
        counts = audit['counts'][name]
        rows.append([label, counts['images'], counts['groups'], f"{counts['normal']} / {counts['pneumonia']}"])
    table(rows, [2.2*inch, .75*inch, 1.1*inch, 1.95*inch])
    p('The internal split reserves every fifth SHA256-ordered proxy group within each class using the salt 20261115. All five roles are disjoint by decoded pixel hash and filename-derived group. These groups are not verified clinical patient identities. The retained test cohort is the same previously inspected 618 images in 426 proxy groups.', 'SmallNote')
    story.append(PageBreak())

    h('Analysis design')
    story.append(Image(str(ROOT / 'output/figures/november_training/protocol_flow.png'), width=6.7*inch, height=6.328*inch))
    p('Figure 1. The primary path uses separate fitting, checkpoint, ensemble-selection, cutoff-fitting and source-test roles. The secondary path partitions only the original training cohort, fitting three seeded members per recipe within each outer fold. All grouping uses filename proxies. These procedures prevent the specified within-run group overlaps but do not establish verified patient independence or erase previous research exposure.', 'SmallNote')
    p('The original source test and the secondary outer folds answer different questions. The source test measures transfer to the inherited test partition, which was already inspected in earlier work. The secondary assessment describes internal performance on the original training population, with outer groups excluded from the current fold\'s fitting decisions. Neither is a new external clinical cohort.')
    story.append(PageBreak())

    h('Training and decision rules')
    p('Both architectures use official torchvision ImageNet V1 initialization [2,3]. A new linear binary head is trained with the full backbone from the first epoch. Inputs are RGB, full-image bicubic letterbox resizes to 320 pixels with ImageNet channel normalization. Training applies affine rotation up to 7 degrees, translation up to 3%, and scale 0.95-1.05 with bilinear sampling. No image crop or test-time augmentation is used.')
    table([['Recipe', 'Architecture', 'Additional regularization', 'Weight decay'],
           ['b0_standard', 'EfficientNet-B0', 'None beyond affine augmentation and model dropout', '0.0001'],
           ['b0_regularized', 'EfficientNet-B0', 'Mixup alpha 0.2; label smoothing 0.05; brightness/contrast jitter 0.15', '0.0005'],
           ['v2s_regularized', 'EfficientNet-V2-S', 'Same additional regularization as b0_regularized', '0.0005']],
          [1.25*inch, 1.25*inch, 2.55*inch, .95*inch])
    p('All recipes use unweighted binary cross-entropy, AdamW with backbone learning rate 0.0001 and head rate 0.001, a 36-epoch cosine schedule to 0.000001, batch size 24, gradient norm clipping at 5, and float16 automatic mixed precision. Evaluation uses float32. Mixup interpolates both inputs and soft targets [4]. The regularized recipes jointly change multiple settings, so their effects are not individually identifiable.')
    p('Seeds are 42, 123 and 2026, with recipe order rotated across seeds. Each saved checkpoint maximizes internal-holdout correct count at cutoff 0.5; tied counts prefer lower log loss. Only complete epochs can qualify. Early stopping requires at least eight epochs and eight consecutive non-improvements unless the time budget intervenes. The 225-minute training budget is divided among remaining trials; unused time is redistributed. Time-limited outcomes are reported, not silently discarded.')
    p('Four predefined equal-weight ensembles are compared: each recipe across its three seeds, and all nine members. Five grouped folds on the 456-image selection cohort fit a maximum-accuracy cutoff on four folds and score the fifth. Pooled correct counts select the ensemble; ties prefer fewer members, then candidate name. This is cutoff cross-fitting, not unbiased nested evaluation of the full research process. The new internal holdout prevents this run from using selection images for checkpoint choice but cannot erase prior experimentation on the source cohorts.')
    p(f"The final raw-score cutoff is {result['threshold']:.10f}, fitted only on the separate calibration cohort ({result['calibration_in_sample_correct']}/482 correct in sample). Candidate cutoffs are zero, midpoints of unique scores and the next representable number above the maximum score; ties choose the lowest. No temperature scaling is applied. The average sigmoid score is not asserted to be a clinical probability. Frozen configuration SHA256: <font size='8'>{escape(result['config_sha256'])}</font>.")
    story.append(PageBreak())
    h('All nine training trials')
    rows = [['Trial', 'Epochs / best', 'Stop reason', 'Internal best correct', 'Selection accuracy at 0.5', 'Minutes']]
    for t in result['trials']:
        rows.append([t['trial'], f"{t['epochs_completed']} / {t['best_epoch']}", t['stop_reason'],
                     f"{t['best_internal_correct']}/781", pct(t['selection_at_0_5']['accuracy']), f"{t['seconds']/60:.2f}"])
    table(rows, [1.7*inch, .65*inch, .9*inch, .9*inch, 1.15*inch, .7*inch])
    p('The selection accuracy in this table uses the fixed cutoff 0.5 for individual models. It is distinct from the candidate ensemble cross-fitted selection metric and from the final calibration-selected cutoff.', 'SmallNote')
    h('Development ensemble comparison')
    rows = [['Candidate', 'Models', 'Cross-fitted correct', 'Cross-fitted accuracy', 'Selection AUROC']]
    for c in choice['candidates']:
        rows.append([c['candidate'], len(c['members']), f"{c['correct']}/456", pct(c['cross_fitted_accuracy']), f"{c['auroc']:.5f}"])
    table(rows, [1.65*inch, .6*inch, 1.2*inch, 1.35*inch, 1.2*inch])
    p('The ranking was frozen before the final source-test inference. These are development estimates after candidate selection, not confirmatory estimates. All candidate scores and fold-specific cutoffs are retained in the machine-readable record.', 'SmallNote')
    story.append(PageBreak())

    h('Learning curves')
    story.append(Image(str(ROOT / 'output/figures/november_training/learning_curves.png'), width=6.8*inch, height=5.829*inch))
    p('Figure 2. Each line is one independently initialized seeded trial; dots show the saved checkpoint. Only complete epochs are plotted. All seeds share the same data partitions. The holdout was used repeatedly for checkpoint choice, so its highest observed accuracy is optimistic. Training objective values are not directly comparable across standard and regularized recipes because mixup and label smoothing change the targets. These curves describe optimization, not external diagnostic performance.', 'SmallNote')
    story.append(PageBreak())

    h('Exploratory source-test results')
    rows = [['Metric', 'Original 320 ensemble', 'November candidate', 'Paired difference (95% interval)']]
    for key, label in [('accuracy', 'Accuracy'), ('sensitivity', 'Sensitivity'), ('specificity', 'Specificity'), ('auroc', 'AUROC'), ('average_precision', 'Average precision'), ('brier_score', 'Brier score')]:
        scale = 100 if key in ['accuracy', 'sensitivity', 'specificity'] else 1
        lo, hi = cmp['paired_cluster_bootstrap_95ci'][key]
        fmt = (lambda v: f'{v*100:.2f}%') if scale == 100 else (lambda v: f'{v:.5f}')
        unit = ' pp' if scale == 100 else ''
        rows.append([label, fmt(cmp['baseline'][key]), fmt(cmp['followup'][key]),
                     f"{scale*cmp['difference'][key]:+.3f}{unit} ({scale*lo:+.3f} to {scale*hi:+.3f})"])
    table(rows, [1.2*inch, 1.3*inch, 1.25*inch, 2.25*inch])
    p(f"Frozen candidate confusion counts are TN {m['tn']}, FP {m['fp']}, FN {m['fn']}, and TP {m['tp']}. The original comparator counts were TN 154, FP 77, FN 1, and TP 386. At the separately prespecified descriptive cutoff 0.5, candidate accuracy was {pct(result['test_at_0_5']['accuracy'])}; that result does not replace the calibration-selected rule.")
    if (HERE / 'primary_frozen_disagreements.json').exists():
        errors = read('primary_frozen_disagreements.json')
        counts = errors['paired_correctness']
        p(f"A post hoc pairing of the frozen decisions found {counts['both_correct']} images correct for both models, {counts['neither_correct']} incorrect for both, {counts['original_only_correct']} correct only for the original, and {counts['november_only_correct']} correct only for November. All {errors['by_inherited_class']['normal']['decisions_differ']} disagreements were NORMAL-labeled images. Both missed the same one pneumonia-labeled image. This descriptive analysis was added after observing outcomes and did not alter either decision rule.", 'SmallNote')
    story.append(Image(str(ROOT / 'output/figures/november_training/test_comparison.png'), width=6.8*inch, height=2.526*inch))
    p('Figure 3. Differences use the same 618 images, with 1,000 paired bootstrap samples of the 426 filename proxy groups (seed 42). Intervals condition on the fitted models and frozen cutoffs. No threshold is refitted during the bootstrap. They omit training, candidate-selection, patient-proxy and external-population uncertainty; no multiplicity adjustment is applied.', 'SmallNote')
    h('Engineering verification and compute')
    p(f"The training region took {result['training_seconds']/60:.2f} minutes, including per-trial evaluation and checkpoint export. The full notebook took {result['total_seconds']/60:.2f} minutes inside the Python program; the Kaggle job duration is recorded separately. One {escape(result['environment']['gpu'])} was used even though two GPUs were allocated. The actual exported command was exercised on two hash-selected test images; all checks passed: {result['cli_all_passed']}.")
    if verification:
        p(f"A separate fresh CPU reproduction returned status <b>{escape(verification['status'])}</b>. It checked all nine checkpoint hashes and strict loads, reconstructed checkpoint epochs, ensemble selection and calibration cutoff, and independently recomputed saved test metrics. For fresh calibration inference, {verification['partitions']['calibration']['changed_predictions']} classifications changed; for fresh test inference, {verification['partitions']['test']['changed_predictions']} changed. Maximum test score difference was {verification['partitions']['test']['max_absolute_score_difference']:.8g} at tolerance 0.0001. This is same-cohort engineering reproduction, not external testing.")
    else:
        p('Full-cohort CPU reproduction has not been completed for this candidate. The two-image command check must not be described as full-cohort verification.')
    story.append(PageBreak())

    if verification and 'descriptive_score_diagnostics' in verification:
        h('Development-to-test score transfer')
        diagnostics = verification['descriptive_score_diagnostics']
        p('The same frozen raw ensemble scores are summarized on the calibration and source-test cohorts. The cutoff was fitted on calibration images, so their classification accuracy is in sample. Fixed ten-bin reliability summaries and class-conditional score quantiles were specified before the current test outcome was observed. They describe transport of the learned score distribution and do not change the model.')
        story.append(Image(str(ROOT / 'output/figures/november_training/score_transfer.png'), width=6.8*inch, height=2.72*inch))
        p('Figure 4. Left: nonempty fixed bins, with larger points indicating more images; the dashed line shows mean score equal to observed pneumonia-label fraction. No interpolation or uncertainty interval is shown. Right: thin segments span 5th-95th percentiles, thick segments 25th-75th, and white circles mark medians. These are distribution summaries, not confidence intervals. Bin counts are retained in the verification record.', 'SmallNote')
        rows = [['Cohort', 'Images', 'Ten-bin ECE', 'Normal mean score', 'Pneumonia mean score']]
        for split in ['calibration', 'test']:
            d = diagnostics[split]
            rows.append([split.title(), d['images'], f"{d['expected_calibration_error']:.5f}",
                         f"{d['class_scores']['normal']['mean_score']:.5f}",
                         f"{d['class_scores']['pneumonia']['mean_score']:.5f}"])
        table(rows, [1.2*inch, .6*inch, 1.1*inch, 1.6*inch, 1.5*inch])
        p('ECE depends on binning and sample size and is not a complete calibration assessment. A difference between cohorts can reflect label mix, acquisition, processing, cohort construction or other unmeasured factors. The available records do not identify the cause. No demographic or site-stratified clinical claim follows from these summaries.')
        story.append(PageBreak())

    if (HERE / 'cv_summary.json').exists():
        cv = read('cv_summary.json'); cv_record = read('cv_run_record.json')
        if cv['status'] != 'completed' or cv_record['status'] != 'completed' or len(cv['training_records']) != 27:
            raise ValueError('Completed grouped-CV execution required for research results')
        cv_verification = read('cv_cpu_verification.json') if (HERE / 'cv_cpu_verification.json').exists() else None
        h('Secondary grouped internal assessment')
        p('Three grouped outer folds cover the original 3,815 training images in 2,153 filename proxy groups. Each outer training cohort is divided again into inner fitting and checkpoint/cutoff groups. The same three recipes and three seeds produce 27 fresh neural models. Outer test images never participate in that fold\'s parameter, checkpoint or cutoff fitting. All 27 checkpoints finish before any outer predictions are generated; equal-weight three-seed fold ensembles use an inner-only maximum-accuracy cutoff.')
        p(f"Actual outer membership was generated by scikit-learn {escape(cv['environment']['sklearn'])}, with shuffle enabled and seed 20261115. The exported manifest and out-of-fold table assignments define this execution. A local preflight using a newer library version produced different assignments, so matching the integer seed alone is not sufficient for exact fold reproduction.", 'SmallNote')
        rows = [['Outer fold', 'Inner fitting images', 'Inner holdout images', 'Outer images / groups']]
        for fold in range(3):
            rec = next(r for r in cv['training_records'] if r['fold'] == fold)
            groups = read('cv_folds.json')[str(fold)]['groups']
            rows.append([fold+1, rec['fit_images'], rec['internal_images'], f"{rec['outer_images']} / {groups}"])
        table(rows, [1.05*inch, 1.5*inch, 1.5*inch, 1.95*inch])
        rows = [['Recipe', 'Correct / 3,815', 'Accuracy', 'Sensitivity', 'Specificity', 'AUROC']]
        for recipe, values in cv['recipes'].items():
            r = values['pooled_metrics']
            rows.append([recipe, values['correct'], pct(r['accuracy']), pct(r['sensitivity']), pct(r['specificity']), f"{r['auroc']:.5f}"])
        if cv_verification:
            g = cv_verification['geometry_control']; r = g['pooled_metrics']
            rows.append(['Image dimensions', g['correct'], pct(r['accuracy']), pct(r['sensitivity']), pct(r['specificity']), f"{r['auroc']:.5f}"])
        table(rows, [1.45*inch, .95*inch, .85*inch, .95*inch, .95*inch, .85*inch])
        story.append(Image(str(ROOT / 'output/figures/november_training/grouped_cv.png'), width=6.5*inch, height=2.91*inch))
        p('Figure 5. Lines connect corresponding outer folds; diamonds show pooled out-of-fold accuracy. They are not independent repetitions or confidence intervals. Fold training sets overlap, and each fold uses a separately fitted cutoff. Pooled AUROC combines differently scaled fold models. These results neither choose a replacement primary model nor establish external generalization.', 'SmallNote')
        story.append(PageBreak())
        h('Cross-validation controls and reproduction')
        p(f"The secondary neural training region took {cv['training_seconds']/60:.2f} minutes. Actual exposure differs across folds, recipes and seeds because of early stopping and the shared 225-minute training cap. All 27 training histories, stop reasons and selected epochs are retained in public records. Complete weights remain in <link href='{cv_record['kaggle_url']}'>Kaggle version {cv_record['kaggle_version']}</link>, subject to account visibility. The complete prediction tables are exported through the independent CPU verification, with exact file hashes checked during capture.")
        if 'retry_of_version' in cv_record:
            p(f"The first commit, version {cv_record['retry_of_version']}, remained queued and was cancelled after zero execution seconds. The completed retry used the exact same submitted source; no training recipe or selection rule changed.", 'SmallNote')
        rows = [['Recipe', 'Fresh fits', 'Complete-epoch range', 'Best-epoch range', 'Time-capped fits', 'Training minutes']]
        for recipe in cv['recipes']:
            trials = [r for r in cv['training_records'] if r['recipe'] == recipe]
            rows.append([recipe, len(trials), f"{min(t['epochs_completed'] for t in trials)}-{max(t['epochs_completed'] for t in trials)}",
                         f"{min(t['best_epoch'] for t in trials)}-{max(t['best_epoch'] for t in trials)}",
                         sum(t['stop_reason'] == 'time_budget' for t in trials), f"{sum(t['seconds'] for t in trials)/60:.2f}"])
        table(rows, [1.4*inch, .65*inch, 1.1*inch, 1.15*inch, 1.1*inch, 1.1*inch])
        p('Time-capped trials retain their best complete-epoch checkpoint. Any partial final epoch remains in the history but cannot qualify for checkpoint selection. Completion of every planned fit does not imply that every model converged.', 'SmallNote')
        if cv_verification:
            p('The dimension-only control uses the natural logarithm of original pixel width and height, a standardized L2 logistic regression (C=1, lbfgs, maximum 1,000 iterations, no class weights), and the same inner/outer partitions. Each scaler and classifier sees only the inner fitting groups. The maximum-accuracy cutoff sees only the inner holdout. All three configurations are frozen before outer scoring, and exported coefficients independently reproduce the fitted probabilities within 0.000000000001. Radiographic pixels are used only to verify image identity, not as predictors.')
            p(f"Independent CPU reconstruction returned <b>{escape(cv_verification['status'])}</b>. It verified all 27 checkpoint hashes, all 3,815 decoded image identities, outer/inner membership, checkpoint epochs, ensemble means and fold cutoffs. It reran all nine fold ensembles on their outer images. Across the nine recipe-fold combinations, {sum(r['changed_predictions'] for r in cv_verification['fold_reports'])} classifications changed; the largest raw-score difference was {max(r['max_absolute_score_difference'] for r in cv_verification['fold_reports']):.8g}, at strict tolerance 0.0001 with zero changed predictions required.")
            rows = [['Recipe vs dimensions', 'Neural only correct', 'Dimensions only correct', 'Accuracy difference']]
            for recipe, values in cv_verification['pooled'].items():
                pairs = values['geometry_paired_correctness']
                rows.append([recipe, pairs['neural_only_correct'], pairs['geometry_only_correct'], f"{100*pairs['accuracy_difference']:+.2f} pp"])
            table(rows, [1.65*inch, 1.35*inch, 1.45*inch, 1.55*inch])
        else:
            p('Independent full-cohort CPU reconstruction and the planned dimension-only comparison have not completed. They are not represented as passed checks or measured control results.')
        p('A dimension-only signal indicates an association between image construction and the source label. It does not prove that the neural networks exploit dimensions, identify a causal mechanism, or establish patient independence. The control was specified after the historical geometry result and while the new primary training was running, before grouped-CV outcomes. This secondary analysis remains exploratory. No fold-based standard error, significance test or model-promotion rule is applied.')
        story.append(PageBreak())

    h('Interpretation and limitations')
    p('The primary result is negative for the requested accuracy objective: development selection did not transfer into higher source-test accuracy. The original benchmark export remains unchanged, and the November candidate is retained as a separate research artifact. Choosing its higher-scoring descriptive test cutoff after seeing the test result would not establish a development-selected improvement.')
    p('This experiment changes training objective, optimization cohort size, checkpoint criterion, regularization and candidate architecture relative to the October benchmark. A difference from the historical model cannot be attributed uniquely to one factor. The standard-versus-regularized B0 comparison is a bundled regularization comparison. The V2-S comparison changes architecture and parameter count. Early stopping and time caps can give trials unequal training exposure.')
    p('All training and evaluation images come from the same inherited source resource. Previous inspection of selection, calibration and test cohorts makes this exploratory even though the present decision rules were recorded prospectively. No test outcome was used to change this run\'s model or cutoff. Filename proxies cannot guarantee patient independence; the earlier perceptual-similarity screen remains incompletely adjudicated. Image geometry previously predicted source labels, raising a possible shortcut association that has not been clinically resolved.')
    p('This analysis used inherited directory classes as its reference labels. It had no verified per-image demographic, acquisition-parameter or clinical-outcome table, and performed no new expert labeling or interreader study. Demographic subgroup performance and label uncertainty were therefore not quantified. The retained test size was inherited from the available benchmark and duplicate-removal rules; no prospective clinical sample-size or power calculation was performed. Public availability of the source images does not establish that these missing clinical details have been verified.')
    if (HERE / 'similarity_reproduction.json').exists():
        p('A separate local reproduction decoded all 5,856 original source images, reconstructed the frozen cohorts, and reproduced the earlier similarity-screen counts: 814 cross-partition candidates, including 269 involving test images. Candidate metadata and pixel hashes are available in the repository. Reproduced aggregate counts do not establish byte-for-byte identity with the unavailable original candidate table.')
        if (HERE / 'test_similarity_visual_review.json').exists():
            visual = read('test_similarity_visual_review.json')
            p(f"A post hoc AI-assisted image-correspondence review inspected all {visual['reviewed_pairs']} test-involving candidates across {visual['reviewed_sheets']} contact sheets. No transformed copy was established at the displayed 540-by-245-pixel panel resolution. Ten pairs overlapped the initial 24-pair review, yielding {visual['unique_pairs_across_both_reviews']} unique candidates viewed and {visual['screen_candidates_not_visually_reviewed']} unreviewed. Pair IDs, panel hashes and observations are retained. This is not independent expert or full-resolution registration-based adjudication; it cannot exclude missed transformations, repeated acquisitions or shared patients. No exclusions or model changes followed.")
        else:
            p('An AI-assisted review of 24 pairs ranked by resized pixel correlation confirmed no duplicate in that selected subset; it was not expert or exhaustive high-resolution adjudication. No cohort exclusions followed this review.')
    p('Accuracy is prevalence dependent and must be read alongside false-positive and false-negative counts. A high or even perfect observed metric on a finite cohort is not a guarantee of future accuracy or sensitivity. Reference labels were inherited, not independently re-read by clinicians. Individual demographics, acquisition details and verified patient identifiers were not available for subgroup or fairness analysis. No clinical deployment, diagnostic claim, prospective trial or external-country performance has been established.')
    p('Independent external testing remains incomplete. The previously investigated external mirror had unresolved release/access statements; no new permission agreement was accepted and no external predictions were produced in this experiment. Claims about external generalization must await an appropriate authorized independent cohort.')
    p('A focused additional review identified VinDr-PCXR/PediCXR as a potential independent pediatric resource, but its files require a signed data-use agreement [6]. A smaller openly licensed collection of mobile-device radiograph photographs was also inspected at the release-page level; reference-label adjudication and patient independence remained unverified [7]. Neither collection was scored. Related work already distinguishes cross-dataset discrimination, calibration and frozen operating-point transfer [8]; the present project is a reproducible exploratory benchmark, not a claim to invent these evaluation principles.')
    story.append(PageBreak())
    h('Reproducibility and availability')
    p(f"The prospective source commit is <link href='https://github.com/bekturovtemirlan000-gif/pneumonia-research-benchmark/commit/{record['prospective_github_commit']}'>GitHub {record['prospective_github_commit'][:12]}</link>. The exact submitted training source hash is <font size='8'>{record['submitted_source_sha256_lf']}</font>. The <link href='{record['kaggle_url']}'>immutable Kaggle training version {record['kaggle_version']}</link> contains full model weights, histories, manifests and prediction files, subject to the notebook's account visibility. The public repository contains source code, captured records, figures and hash-checked public prediction exports; it does not imply public access to the Kaggle weight archive.")
    if (HERE / 'primary_prediction_audit.json').exists():
        p('Complete calibration and test CSV exports were reconstructed from bounded JSON log blocks and matched to their exact Kaggle file hashes. A separate local audit recomputed their metrics, frozen classifications, score-distribution summaries and CPU differences using round-trip float parsing. Corresponding grouped-CV exports, when completed, contain one out-of-fold score per original training image for each neural recipe and the dimension-only comparator. These exports provide image-level numerical reproducibility without redistributing image pixels or weights.')
    if (HERE / 'comparator_export.json').exists():
        p('The original 320-pixel comparator\'s complete saved test scores were exported separately with their frozen temperature scaling and cutoff. The local audit paired all 618 image identities and reproduced every reported paired bootstrap interval using 1,000 filename-group resamples, seed 42. This is retrospective verification of the already-recorded comparison; it performs no new fitting or model selection.')
    p('Run the exported model with model.py --model-dir MODEL_FOLDER --image IMAGE_PATH --device cpu, keeping the configuration beside its checkpoint subdirectories. The loader verifies checkpoint hashes. The GitHub notebook embeds source modules so it can be rerun with the original dataset and benchmark inputs attached. Fresh reruns may differ across hardware or software versions.')
    p('The report distinguishes data roles, reference-label limitations, reproducibility and external validity in line with the topics emphasized by CLAIM [5]; it does not claim complete checklist compliance or clinical readiness.', 'SmallNote')
    p('Code development, analysis checks and drafting were assisted by OpenAI Codex. Human review of the manuscript and author declarations remains pending. Author affiliation, funding, conflicts and the applicable ethics determination have not been supplied and are not invented here. No journal submission or peer review is claimed.', 'SmallNote')
    h('References')
    refs = [
        ('Kermany DS et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172:1122-1131.e9.', 'https://doi.org/10.1016/j.cell.2018.02.010'),
        ('Tan M, Le Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML. 2019;97:6105-6114.', 'https://proceedings.mlr.press/v97/tan19a.html'),
        ('Tan M, Le Q. EfficientNetV2: Smaller Models and Faster Training. ICML. 2021;139:10096-10106.', 'https://proceedings.mlr.press/v139/tan21a.html'),
        ('Zhang H, Cisse M, Dauphin YN, Lopez-Paz D. mixup: Beyond Empirical Risk Minimization. ICLR. 2018.', 'https://openreview.net/forum?id=r1Ddp1-Rb'),
        ('Tejani AS et al. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): 2024 Update. Radiol Artif Intell. 2024;6:e240300.', 'https://doi.org/10.1148/ryai.240300'),
        ('Pham HH, Tran TT, Nguyen HQ. VinDr-PCXR pediatric chest X-ray dataset. PhysioNet. Version 1.0.0; 2022. Access page reviewed 14 November 2026 UTC.', 'https://physionet.org/content/vindr-pcxr/1.0.0/'),
        ('Raihan et al. Child Pneumonia Dataset. Mendeley Data. Version 1; 2025. Release-page review only.', 'https://data.mendeley.com/datasets/3tx8xymdsv/1'),
        ('Nazim-E-Alam. Cross-dataset transportability of pediatric chest X-ray deep learning across three countries: discrimination, calibration, operating-point failure, and limited-label recovery. arXiv:2609.05140v1; 2026. Preprint; abstract-level review.', 'https://arxiv.org/abs/2609.05140v1')]
    for i, (text, url) in enumerate(refs, 1): p(f"{i}. {escape(text)} <link href='{url}' color='#175e81'>Source</link>", 'SmallNote')
    def footer(canvas, doc):
        canvas.saveState(); canvas.setFont('Helvetica', 8); canvas.setFillColor(colors.HexColor('#536a75'))
        canvas.drawString(.75*inch, .43*inch, 'Pneumonia research | November training addendum | Exploratory')
        canvas.drawRightString(7.75*inch, .43*inch, str(doc.page)); canvas.restoreState()
    SimpleDocTemplate(str(output), pagesize=(8.5*inch, 11*inch), rightMargin=.75*inch, leftMargin=.75*inch,
                      topMargin=.65*inch, bottomMargin=.65*inch,
                      title='Development-accuracy selection after regularized pneumonia-model training').build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)
    return output


if __name__ == '__main__': build()
