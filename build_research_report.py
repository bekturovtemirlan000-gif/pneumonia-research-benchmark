"""Build an evidence-linked research report using the bundled document runtime."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, Image


def build(output, benchmark=None):
    root=Path(__file__).resolve().parent
    baseline=json.loads((root/'results/baseline_metrics.json').read_text())
    followup=json.loads((root/'results/followup_metrics.json').read_text())
    result_path=benchmark/'metrics.json' if benchmark else root/'results/benchmark224_metrics.json'
    trial_path=benchmark/'all_trial_results.json' if benchmark else root/'results/benchmark224_all_trial_results.json'
    result=json.loads(result_path.read_text()) if result_path.is_file() else None
    resolution=json.loads((root/'results/benchmark320_metrics.json').read_text())
    resolution_uncertainty=json.loads((root/'results/resolution_paired_uncertainty.json').read_text())
    styles=getSampleStyleSheet()
    styles['Normal'].fontName='Times-Roman'; styles['Normal'].fontSize=11
    styles['Normal'].leading=14; styles['Normal'].spaceAfter=7
    styles['Normal'].allowWidows=0; styles['Normal'].allowOrphans=0
    for name in ['Title','Heading1','Heading2']:
        styles[name].textColor=colors.black
        styles[name].fontName='Times-Bold'
    styles['Title'].fontSize=22; styles['Title'].leading=25
    styles['Heading1'].fontSize=14; styles['Heading1'].spaceBefore=13
    styles['Heading1'].spaceAfter=6
    styles['Heading1'].keepWithNext=True
    styles.add(ParagraphStyle(name='TableText',fontName='Times-Roman',fontSize=9,leading=12))
    styles.add(ParagraphStyle(name='Small',fontName='Times-Roman',fontSize=9,leading=12,spaceAfter=5))
    story=[]
    def p(text,style='Normal'): story.append(Paragraph(text,styles[style]))
    def h(text): p(text,'Heading1')
    def table(rows,widths):
        content=[[Paragraph(escape(str(cell)),styles['TableText']) for cell in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9edf1')),
             ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
             ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
             ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7)]))
        story.append(KeepTogether([t,Spacer(1,9)]))
    p('Architecture, resolution and calibration in pediatric pneumonia image classification','Title')
    p('An exploratory reproducible benchmark','Heading2')
    p('Research report | 22 October 2026','Small')
    h('Abstract')
    p('<b>Purpose.</b> To compare conventional pretrained image classifiers under a fixed development protocol, separating ranking performance from calibration and threshold-dependent classification. This report concerns dataset labels on pediatric chest radiographs and does not establish clinical diagnosis or utility.')
    p('<b>Methods.</b> A decoded-image audit and conservative filename grouping retain 3,815 training, 456 model-selection, 482 calibration and 618 test images. EfficientNet-B0, DenseNet121 and ResNet18 are trained with three seeds each at 224 and 320 pixels (18 trials). Mean selection AUROC chooses the architecture; its seed probabilities are averaged. A positive temperature and a Youden threshold are fit on the separate calibration partition before test evaluation. All comparisons on the original test set are exploratory because prior results were inspected.')
    if result:
        m=resolution['test']
        p(f"<b>Results.</b> All 18 trials completed. Development-only selection across both resolutions chose the 320-pixel EfficientNet-B0 ensemble. Its test AUROC was {m['auroc']:.4f} (95% group-bootstrap interval 0.9470 to 0.9787), accuracy {100*m['accuracy']:.2f}%, sensitivity {100*m['sensitivity']:.2f}% and specificity {100*m['specificity']:.2f}%. The paired AUROC difference versus the 224-pixel ResNet18 ensemble was 0.0059 (95% interval -0.0055 to 0.0170), which does not establish improvement. CPU test reproduction changed zero classifications for either ensemble. A post hoc classifier using only source image dimensions achieved test AUROC 0.9084, highlighting a dataset association requiring further investigation.")
    else:
        p('<b>Status.</b> The nine-trial benchmark is running. Its outcomes are not yet available. Earlier EfficientNet experiments demonstrate why a change in accuracy alone is insufficient evidence of improvement.')
    p('<b>Interpretation.</b> The study is an exploratory benchmark with explicit audit and reproducibility checks. External testing, independently verified patient identities and clinical utility remain unestablished. A longer training duration alone does not resolve these limitations.')
    h('Introduction')
    p('Pediatric pneumonia image classification is frequently studied using the Guangzhou chest-radiograph resource distributed by Kermany and colleagues [1]. Performance on one source dataset does not demonstrate transportability. Zech and colleagues reported variable pneumonia-model generalization across datasets [2], and a recent pediatric preprint explicitly examined discrimination, calibration and fixed-threshold transport across countries [3]. These findings motivate reporting several complementary outcomes rather than treating accuracy as a complete measure of model quality.')
    p('This project asks how architecture, input resolution and random initialization affect discrimination and the behavior of a development-selected decision rule under a constrained compute budget. It compares three established architectures, not a newly invented network. Temperature scaling is an established calibration technique [4]. The intended contribution is a transparent, reproducible benchmark and an honest account of its limitations; novelty sufficient for a particular publication venue has not been established.')
    h('Materials and methods')
    environment_path=root/'results/benchmark224_environment.json'
    if environment_path.is_file():
        env=json.loads(environment_path.read_text())
        p(f"<b>Computing environment.</b> Both resolution arms ran sequentially on one {escape(env['gpu'])}, with PyTorch {escape(env['torch'])}, torchvision {escape(env['torchvision'])}, NumPy {escape(env['numpy'])}, pandas {escape(env['pandas'])} and scikit-learn {escape(env['sklearn'])}. The local CPU development environment differs; its dependency snapshot is recorded separately. GPU and CPU agreement is evaluated numerically rather than assumed to be bitwise identical.")
    p('<b>Study design.</b> This is a retrospective computational study of public research images. No patient care is involved. The preliminary baseline and follow-up were completed before this benchmark was designed. Their original test outcomes were already known; therefore the current source test evaluation cannot be described as an untouched confirmatory test.')
    p('<b>Source population and reference labels.</b> The source authors describe retrospective anterior-posterior radiographs from children aged one to five at Guangzhou Women and Children&apos;s Medical Center, obtained during routine care [9]. They report excluding unreadable or low-quality scans, diagnosis grading by two expert physicians, and evaluation-set review by a third. These are source-reported procedures. This reanalysis inherits directory labels without independent clinical adjudication; individual demographics, microbiological evidence and verified clinical patient IDs were not available to this analysis. OCT-specific labeling and independence statements in the shared resource are not treated as chest-radiograph evidence.')
    p('<b>Data audit.</b> The attached Kaggle resource contains 5,856 decoded images. The audit hashes grayscale dimensions and pixels, detects conflicting labels for identical images, removes exact duplicate test images, and excludes development images sharing a derived group or pixel hash with test. Development duplicates are removed deterministically. The audit removes 479 development images in aggregate and six exact duplicate test images, leaving 5,371 images. The aggregate development count combines overlapping exclusion reasons and is not a count of unique patients. Near-duplicate detection and clinical patient-ID verification are not established by this exact-hash check.')
    p('<b>Grouping limitation.</b> Groups are derived from filename patterns IM, NORMAL2-IM and person-number. The person-number rule groups bacterial and viral filenames together, although numbers may be reused across disease namespaces. This conservative grouping may merge unrelated children and exclude additional development images. The groups should not be called verified patient identities. The frozen split is retained to preserve the planned comparison.')
    table([['Partition','Images','Derived groups','Normal','Pneumonia'],
           ['Training',3815,2153,1078,2737],['Model selection',456,270,137,319],
           ['Calibration',482,268,133,349],['Source test',618,426,231,387]],
          [1.5*inch,1*inch,1.2*inch,1*inch,1.3*inch])
    p('The earlier 938-image validation cohort is divided by group and class into selection and calibration partitions using a fixed hash ordering. Explicit original held-out group membership and SHA-256 cohort digests prevent scikit-learn version changes from silently changing the training or test images. Every pair of partitions must have zero exact-pixel-hash and derived-group overlap.')
    p('<b>Image processing.</b> Images are converted to RGB, resized with bicubic interpolation and padded to 224 by 224 or 320 by 320 pixels while preserving the entire image and aspect ratio. Training applies random affine augmentation with at most 7 degrees of rotation, translation fractions 0.03 and scale 0.95 to 1.05. Tensors use ImageNet channel means and standard deviations. Lossless PNG caching of resized images avoids repeated original-image decoding. Local checks confirm equality of cached and direct preprocessing tensors.')
    p('<b>Model training.</b> Torchvision ImageNet V1 weights [7] initialize EfficientNet-B0, DenseNet121 and ResNet18; each receives one binary output. Seeds are 42, 123 and 2026, including each trial\'s data-loader order and augmentation seeds. AdamW uses a backbone learning rate of 0.0001, head learning rate of 0.001 and weight decay of 0.0001. Weighted binary cross entropy assigns the positive class the training normal-to-pneumonia count ratio. Batch size is 32. Training uses mixed precision, whereas evaluation uses float32.')
    p('The scheduler halves the learning rate after a plateau of two epochs. Selection AUROC determines checkpoint replacement with a minimum improvement of 0.00001; training stops after eight unimproved epochs or a time limit, with a maximum of 60 epochs per trial. Architecture order rotates across seeds. The 270-minute audit-and-training budget is allocated dynamically over remaining trials. This is a shared time-budget comparison, not a guarantee of equal epochs or identical FLOPs. Partial final epochs and stop reasons are retained in the histories. One GPU is used even if Kaggle allocates a T4 pair.')
    p('<b>Selection and calibration.</b> Architecture is selected by mean model-selection AUROC over all three seeds; exact ties use alphabetical order. Seed sigmoid probabilities are averaged. The averaged probability is clipped to [0.0000001, 0.9999999], converted to a logit and divided by a positive scalar temperature. Temperature minimizes calibration-set negative log likelihood, with log-temperature bounded between -3 and 3. Maximum Youden J on calibration scores determines the binary cutoff. The inference configuration is saved before test evaluation. No test-based choice of model, temperature or threshold is permitted.')
    p('<b>Outcome measures.</b> Primary descriptive outcomes are AUROC and sensitivity and specificity at the frozen development threshold. Average precision, accuracy, Brier score, log loss and confusion counts supplement them. Fixed raw-score threshold 0.5 results are also retained. Positive temperature scaling preserves ordering apart from clipping or numerical ties; an AUROC gain is not an intended calibration effect. Calibration and threshold fitting share one calibration partition, so in-sample calibration performance should not be interpreted as unbiased future performance.')
    p('<b>Uncertainty and verification.</b> Paired comparisons retain image-hash alignment and resample filename-derived groups with replacement. Percentile intervals describe source test sampling conditional on the fitted models; they omit training and model-selection uncertainty. Three seeds provide limited descriptive evidence of training variation. No multiplicity-adjusted confirmatory claims are planned. Artifact checks reconstruct the selected ensemble from member scores, recompute metrics, and verify split identity and exported checkpoint paths.')
    p('<b>Resolution extension.</b> Before the new benchmark test results were observed, an exploratory 320-pixel arm was specified with the same nine architecture-seed combinations and frozen cohorts. It runs sequentially after the 224-pixel benchmark, with a 180-minute cap within the combined compute budget. Each arm selects its architecture independently from development data. A combined choice, if reported, uses mean selection AUROC only, breaking ties by lower resolution then alphabetical architecture. All arms remain reportable; expanded selection can increase optimism. Both arms completed, with Kaggle job runtimes of 3,048.6 and 5,026.1 seconds (134.58 minutes combined, including job overhead). All 18 trials stopped by the early-stopping rule rather than exhausting their caps. Training duration was not extended after convergence merely to fill the five-hour work session.')
    p('<b>Similarity screening.</b> A separate CPU audit compares 64-bit low-frequency DCT perceptual hashes across partitions, with the DC bit zero and a Hamming-distance cutoff of six. This is a candidate screen, not validated duplicate detection. It is not used to alter the frozen experiment cohorts.')
    h('Results')
    p('The completed preliminary baseline and follow-up used the earlier shared validation partition for checkpoint and threshold selection. They provide project context, not a controlled estimate of the effects of the new separated calibration design. The follow-up selected a single seed-123 model rather than its averaged ensemble.')
    rows=[['Source test metric','Baseline','Selected follow-up']]
    for label,key,percent in [('AUROC','auroc',False),('Average precision','average_precision',False),('Accuracy','accuracy',True),('Sensitivity','sensitivity',True),('Specificity','specificity',True)]:
        values=[baseline['test'][key],followup['test'][key]]
        rows.append([label]+[f'{100*v:.2f}%' if percent else f'{v:.5f}' for v in values])
    table(rows,[2.6*inch,1.7*inch,1.7*inch])
    p('Accuracy increased from 80.42% to 88.19% under the separately selected thresholds, but AUROC decreased from 0.97061 to 0.96844. Sensitivity decreased from 100.00% to 99.22%, while specificity increased from 47.62% to 69.70%. Brier score worsened from 0.13507 to 0.14308. At a common raw threshold of 0.5, accuracy was 84.47% and 84.30%, respectively. These observations do not support an across-the-board improvement claim.')
    if result and trial_path.is_file():
        trials=json.loads(trial_path.read_text())
        p('<b>224-pixel individual trials.</b> The following values describe individual calibrated models, not their ensemble.')
        rows=[['Architecture and seed','Epochs','Selection AUROC','Test AUROC','Test Brier']]
        for r in trials:
            rows.append([r['trial'],r['epochs_completed'],f"{r['selection_auroc']:.5f}",f"{r['calibrated_test']['auroc']:.5f}",f"{r['calibrated_test']['brier_score']:.5f}"])
        table(rows,[2.3*inch,.7*inch,1*inch,1*inch,1*inch])
        figure=root/'output/figures/benchmark224/trial_discrimination_and_calibration.png'
        if figure.is_file():
            figure_image=Image(str(figure),width=6.6*inch,height=2.475*inch)
            figure_image.keepWithNext=True
            story.append(figure_image)
            p('Figure 1. All nine 224-pixel trials. Colors identify architectures; symbols identify seeds. Horizontal segments in A and B show the arithmetic mean of individual-trial AUROCs, not ensemble AUROC or uncertainty intervals. C connects each trial\'s raw and calibrated Brier score; the black dashed line is the separately calibrated, development-selected ensemble. All test results are exploratory.','Small')
        p('All trial-level test values are reported after development-only selection. They must not be used to retroactively replace the selected architecture. The original Kaggle outputs are authoritative. These compact results were transcribed programmatically from the fully expanded visible JSON preview, with field-count and confusion-count checks; the complete new archive has not yet been downloaded locally.')
        p('The selected ResNet18 ensemble used temperature 0.67293 and cutoff 0.72091. Its confusion counts were TN 154, FP 77, FN 1 and TP 386. Calibration worsened source-test Brier score from 0.12752 to 0.13105 and log loss from 0.97988 to 1.27754. Its test AUROC is lower than either earlier selected experiment. Thus, this comparison does not establish that more architectures, separate calibration, or additional training improved general performance.')
        comparison_path=root/'results/benchmark224_paired_uncertainty.json'
        if comparison_path.is_file():
            uncertainty=json.loads(comparison_path.read_text()); paired=uncertainty['paired_comparison']
            rows=[['Metric','224-pixel ensemble (95% interval)','Difference versus prior follow-up (95% paired interval)']]
            for label,key,is_percent in [('AUROC','auroc',False),('Average precision','average_precision',False),('Accuracy','accuracy',True),('Sensitivity','sensitivity',True),('Specificity','specificity',True),('Brier score','brier_score',False)]:
                scale=100 if is_percent else 1
                value=uncertainty['benchmark_metrics'][key]*scale
                lo,hi=[v*scale for v in uncertainty['benchmark_cluster_bootstrap_95ci'][key]]
                delta=paired['difference'][key]*scale
                dlo,dhi=[v*scale for v in paired['paired_cluster_bootstrap_95ci'][key]]
                fmt='.2f' if is_percent else '.4f'
                rows.append([label, f'{value:{fmt}}'+('%' if is_percent else '')+f' [{lo:{fmt}}, {hi:{fmt}}]',
                             f'{delta:{fmt}}'+(' pp' if is_percent else '')+f' [{dlo:{fmt}}, {dhi:{fmt}}]'])
            table(rows,[1.2*inch,2.4*inch,2.4*inch])
            p('Intervals use 1,000 paired filename-group bootstrap replicates over 426 groups and 618 identical test images. Differences are the new ResNet18 ensemble minus the earlier selected EfficientNet-B0 seed-123 model. Percentage-point differences are marked pp. AUROC and accuracy intervals span zero, which neither proves equivalence nor establishes improvement. The Brier interval is negative, but the comparison is exploratory and unadjusted for multiplicity; these intervals omit training and model-selection uncertainty. Kaggle analysis version 360892651 completed successfully.')
        p('Calibration-only group resampling (1,000 replicates, 268 filename groups) yielded threshold percentiles 2.5/25/50/75/97.5 of 0.18496/0.72091/0.72091/0.73699/0.94266. The fitted temperature was held fixed, so temperature-fitting uncertainty is omitted. The deployed cutoff was unchanged.')
        verification_path=root/'results/benchmark224_cpu_verification.json'
        if verification_path.is_file():
            verification=json.loads(verification_path.read_text())
            p(f"Independent CPU reproduction {escape(verification['status'])}. All nine checkpoints loaded strictly and passed a forward check. Decoded-pixel identity was verified for all {verification['images']} test images. Fresh ensemble inference changed {verification['changed_predictions']} classifications; the maximum calibrated-score difference was {verification['max_absolute_score_difference']:.8f}, below the prespecified 0.0001 tolerance. This establishes engineering reproduction of the saved model on the same source cohort, not independent clinical or external validation. Kaggle verification version 360888922 completed in 386.9 seconds.")
    else:
        p('The new benchmark has not completed. No trial or ensemble outcome is imputed from the preliminary experiments. Final comparisons, uncertainty intervals and trained-model verification will be added only after the exports pass the integrity checks.')
    h('Resolution comparison')
    high_trials=json.loads((root/'results/benchmark320_all_trial_results.json').read_text())
    p('<b>320-pixel individual trials.</b> All nine planned trials completed. The same architecture-seed combinations and image identities were retained across resolutions. These are matched protocol comparisons, with adaptive early stopping; they do not isolate resolution from differences in selected checkpoints or training duration.')
    rows=[['Architecture and seed','Epochs','Selection AUROC','Test AUROC','Test Brier']]
    for r in high_trials:
        rows.append([r['trial'],r['epochs_completed'],f"{r['selection_auroc']:.5f}",f"{r['calibrated_test']['auroc']:.5f}",f"{r['calibrated_test']['brier_score']:.5f}"])
    table(rows,[2.3*inch,.7*inch,1*inch,1*inch,1*inch])
    figure=root/'output/figures/resolution_comparison/matched_resolution_trials.png'
    figure_image=Image(str(figure),width=6.6*inch,height=6.6*inch*1152/3120)
    figure_image.keepWithNext=True
    story.append(figure_image)
    p('Figure 2. All 18 trials, paired by architecture and seed across resolutions. Lines join individual fitted models; they are not confidence intervals or ensemble results. A and B share an AUROC scale. C shows calibrated test Brier scores (lower is better). Selection and source-test rankings differ, so retrospective choice of the best test result would violate the protocol.','Small')
    p('The combined development-only selector chose the 320-pixel EfficientNet-B0 ensemble, whose mean seed selection AUROC was 0.999085, versus 0.998566 for the 224-pixel ResNet18 ensemble. Its temperature was 1.34056 and frozen cutoff 0.83170. Test confusion counts were TN 154, FP 77, FN 1 and TP 386. These aggregate counts match the 224-pixel ensemble, but the two models do not make identical image-level predictions.')
    u=resolution_uncertainty; paired=u['paired_comparison']
    rows=[['Metric','320-pixel ensemble (95% interval)','320 minus 224 (95% paired interval)']]
    for label,key,is_percent in [('AUROC','auroc',False),('Average precision','average_precision',False),('Accuracy','accuracy',True),('Sensitivity','sensitivity',True),('Specificity','specificity',True),('Brier score','brier_score',False)]:
        scale=100 if is_percent else 1; fmt='.2f' if is_percent else '.4f'
        value=u['benchmark_metrics'][key]*scale
        lo,hi=[v*scale for v in u['benchmark_cluster_bootstrap_95ci'][key]]
        delta=paired['difference'][key]*scale
        dlo,dhi=[v*scale for v in paired['paired_cluster_bootstrap_95ci'][key]]
        rows.append([label,f'{value:{fmt}}'+('%' if is_percent else '')+f' [{lo:{fmt}}, {hi:{fmt}}]',f'{delta:{fmt}}'+(' pp' if is_percent else '')+f' [{dlo:{fmt}}, {dhi:{fmt}}]'])
    table(rows,[1.2*inch,2.4*inch,2.4*inch])
    p('The paired analysis used 1,000 group-bootstrap replicates on the same 618 images and 426 filename groups. All paired intervals include zero; this neither establishes improvement nor proves equivalence. The zero-width sensitivity-difference interval reflects identical observed decisions among pneumonia-labeled images in this sample, not absence of uncertainty in future sensitivity. No multiplicity correction or retraining was performed. The 320-pixel ensemble had a higher test Brier score than the 224-pixel ensemble (0.13948 versus 0.13105), despite its higher AUROC point estimate.')
    p('Within the 320-pixel ensemble, calibration reduced test Brier score from 0.14313 to 0.13948 and log loss from 1.15973 to 0.78876. Calibration therefore behaved differently across arms: it worsened these source-test measures for the 224-pixel ensemble. Raw-score threshold 0.5 gave the 320-pixel ensemble 84.79% accuracy, 100.00% sensitivity and 59.31% specificity. Calibration scores are not established patient-level disease probabilities.')
    p('An exploratory development-selection stability analysis was specified after the 224-pixel test result was observed and before the 320-pixel test result. It resampled the 270 selection groups jointly across six architecture-resolution candidates, averaging their three fixed seed AUROCs. The frozen 320-pixel EfficientNet-B0 choice won 852 of 1,000 replicates; 224-pixel ResNet18 won 112, 320-pixel ResNet18 won 20 and 224-pixel DenseNet121 won 16. Nine replicates had tied maxima resolved by the prespecified ordering. These frequencies describe selection-sample sensitivity; they are not posterior probabilities or evidence of external superiority. Checkpoint selection used this same partition, and no retraining occurred, so selection optimism remains. The chosen model was not changed.')
    p('Calibration-only threshold resampling for the 320-pixel ensemble yielded 2.5/25/50/75/97.5 percentiles of 0.60672/0.60672/0.83170/0.84011/0.89223 across 1,000 replicates, with eight distinct thresholds. Temperature was held fixed and the original cutoff retained. This descriptive spread is not a confidence interval for a clinically optimal threshold.')
    verification=json.loads((root/'results/benchmark320_cpu_verification.json').read_text())
    p(f"The 320-pixel CPU verification passed in 433.3 seconds (Kaggle version 360906861). All nine checkpoints loaded strictly and passed a forward check. Decoded-pixel identities matched for all 618 test images. Fresh selected-ensemble inference changed {verification['changed_predictions']} classifications; maximum score difference was {verification['max_absolute_score_difference']:.8f}, below tolerance 0.0001. Artifact validation independently reconstructed the ensemble and all reported trial test metrics. The paired analysis completed in 66.2 seconds (version 360907021). These are engineering checks on the same images, not external validation.")
    p('Post hoc frozen-error analysis (Kaggle version 360916945) found 531 images classified correctly by both ensembles and 69 incorrectly by both. Each ensemble alone classified nine additional images correctly. All 18 disagreements were NORMAL-labeled: 145 normal images were correctly classified by both and 68 were false positives for both. Both ensembles detected the same 386 of 387 pneumonia-labeled images. Among source filenames, 239 of 240 bacterial-labeled and all 147 viral-labeled pneumonia images were detected by each model. These are descriptive binary-classifier counts, not evidence of bacterial-versus-viral diagnostic performance or guaranteed future sensitivity. Filename annotations were not independently confirmed. One proxy group spans both pneumonia filename subgroups; subgroup group counts must not be summed as independent patients. No model or cutoff was changed.')
    calibration_figure_start=len(story)
    h('Frozen-score calibration diagnostics')
    figure=root/'output/figures/calibration_diagnostics/frozen_reliability.png'
    figure_image=Image(str(figure),width=6.6*inch,height=4.2*inch)
    figure_image.keepWithNext=True
    story.append(figure_image)
    p('Figure 3. Reliability points and image counts in ten equal-width score bins, using saved calibrated predictions. Empty bins are omitted from the upper panels; lower panels retain all bin positions. The calibration partition has 482 images and the test partition 618. Small bins are unstable, and the partition used for fitting is optimistic. These curves do not establish patient-level probability calibration or isolate a cause of the partition difference.','Small')
    story[calibration_figure_start:]=[KeepTogether(story[calibration_figure_start:])]
    p('The ten-bin expected calibration error was 0.00796 on calibration versus 0.14583 on test for the 224-pixel ensemble, and 0.00941 versus 0.16577 for the 320-pixel ensemble. This bin-dependent descriptive measure has no uncertainty interval here. For the selected 320-pixel model, the mean score on NORMAL-labeled images increased from 0.02119 in calibration to 0.44694 in test. Sparse intermediate bins and many high-scoring normal images make the limits of development calibration visible. These are post hoc diagnostics; no model, temperature or cutoff was adjusted.')
    p('Metric convention matters: the saved diagnostic script clips calibrated scores to [0.0000001, 0.9999999] before log loss, whereas the main benchmark passes float64 calibrated scores directly to scikit-learn with its default numerical clipping. The 224-pixel diagnostic test log loss is therefore 1.16079 rather than the main benchmark value 1.27754. The main results retain the original benchmark convention; diagnostic values are labeled separately. Brier scores and reliability-bin values do not use this additional log-loss clipping.')
    h('Post hoc image-dimension control')
    p('After inspecting the neural-network outcomes, a fixed diagnostic control was specified using only natural-log source JPEG width and height. Standardization and an L2 logistic regression (C=1, unweighted classes, maximum 1,000 iterations) were fit on the 3,815 training images. The cutoff remained 0.5, with no hyperparameter search or temperature fitting. Decoded pixel hashes verified the identity of all 5,371 images but were not predictor features. The control and paired-analysis plans were recorded before its outcomes were observed; this timing does not make the analysis confirmatory.')
    geometry=json.loads((root/'results/geometry_control.json').read_text(encoding='utf-8'))
    rows=[['Partition','Dimension-only AUROC','Accuracy','Brier score']]
    for split in ('train','selection','calibration','test'):
        g=geometry['metrics'][split]
        rows.append([split.title(),f"{g['auroc']:.4f}",f"{100*g['accuracy']:.2f}%",f"{g['brier_score']:.4f}"])
    table(rows,[1.4*inch,1.8*inch,1.4*inch,1.4*inch])
    p('On 618 test images, the dimension-only AUROC was 0.9084 (95% filename-group bootstrap interval 0.8807 to 0.9343), sensitivity 96.38% and specificity 54.98%. Confusion counts were TN 127, FP 104, FN 14 and TP 373. All 5,371 exported scores reconstructed exactly from the saved dimensions and coefficients, with zero changed classifications. The control job completed in 1,221.4 seconds and its separate CPU verification and paired analysis in 33.5 seconds.')
    p('In 1,000 paired resamples of the same 426 test groups, the geometry-minus-neural AUROC difference was -0.0556 (95% interval -0.0798 to -0.0325), and the accuracy difference was -6.47 percentage points (interval -9.15 to -3.88). The Brier difference was -0.0062 (interval -0.0223 to 0.0106). The neural model ranked labels better in this exploratory source comparison, but its Brier score was not clearly better. These intervals are conditional on the fitted models and omit training, selection and multiplicity uncertainty.')
    p('Strong label predictability from source dimensions identifies a dataset association that may reflect acquisition, processing or cohort composition. Width and height are pixel dimensions, not physical anatomical measurements. This experiment does not show that the neural networks use dimensions, identify the causal source of the association, or establish performance on another hospital. No neural checkpoint, cohort or threshold was changed following these findings.')
    h('Discussion and limitations')
    p('The preliminary results illustrate that operating thresholds can change observed accuracy and specificity without improving ranking or calibration. A credible analysis must therefore distinguish performance of the score from performance of a particular decision rule. The new experiment separates model selection from calibration, but its small calibration cohort and conservative grouping introduce uncertainty that additional GPU time cannot remove.')
    p('The source test set was previously inspected. Its results are exploratory, the filename groups are proxies, and the study lacks independent clinical adjudication, documented acquisition-site strata and adequate demographic information for a fairness analysis. Exact-pixel duplicate removal cannot rule out transformed duplicates, pretraining overlap or acquisition shortcuts. The binary dataset target does not establish etiology, severity, treatment need or validity for non-chest images.')
    p('The similarity screen found 814 candidate cross-partition pairs, including 269 involving test images, among 5,371 retained images. Limited contact-sheet inspection showed visible differences in reviewed pairs and did not confirm image duplication. Most candidates remain unreviewed. Perceptual similarity can reflect shared anatomy, and visual differences do not establish different patients. These counts must not be presented as a measured leakage rate or proof of clean patient separation.')
    p('BDCXR-3257 is a candidate external pediatric source [5]. The public record lists CC BY 4.0 and describes contrast enhancement and cropping, which would create preprocessing differences. A documentation-only inspection of a Kaggle copy [8] confirmed 880 normal, 1,477 bacterial and 900 viral images and 3,257 metadata rows. Filename, Age, Gender and Class fields have no missing values. Included certificates agree with these counts, resolving the numerical discrepancy in the public description; no individual patient identifier is supplied.')
    p('The mirror provides no provenance citation and lists its license as unknown. Its included agreement identifies a particular hospital and university research laboratory as parties, limits access to authorized research personnel, and restricts raw-image redistribution. These statements have not been reconciled with the public CC BY listing. Document authenticity and release authority were not independently verified. No external inference was performed, no agreement was accepted, and no claim of applicable ethics approval is made for the present project. External testing remains pending source clarification.')
    p('Related pediatric transport work already evaluates BDCXR and VinDr [3]. Differences in development cohorts, preprocessing and model design prevent direct claims of superiority over that study. This report is not a clinical validation or evidence of readiness for deployment. Reporting guidance such as CLAIM 2024 [6] informs transparency; merely referencing the checklist is not certification of completeness.')
    h('Reproducibility and publication status')
    p('The repository contains training source, Kaggle notebooks, frozen cohort rules, protocol JSON, inference code and artifact validators. Kaggle versions 360875315 and 360888746 are the completed 224- and 320-pixel training runs. Earlier versions 360813730 and 360829588 contain the completed baseline and follow-up. The submitted source hash, runtime versions, split manifest, histories, checkpoints and per-image predictions should accompany the final analysis. Data and weights are excluded from ordinary source control.')
    p('The compact-record audit cross-checks trial selections, confusion-derived measures, paired differences and bootstrap winner counts. It does not independently recompute image-level AUROC or uncertainty intervals. Original GPU raw calibration probabilities were not exported. A separate CPU run (Kaggle version 360912398) evaluated all 482 calibration images for each selected ensemble, exported fresh raw scores and independently refit temperature and cutoff. Relative temperature differences were below 0.000002 and absolute cutoff differences below 0.0000014. The selected 320-pixel ensemble passed all checks. The 224-pixel comparator changed one calibration classification, so the combined run correctly failed its strict zero-change criterion after saving both reports.')
    p('Boundary inspection (version 360914770) found that this comparator score originally equaled its cutoff exactly; CPU inference moved it lower by 0.000001038. Reading saved scores with round-trip precision reproduced all stored original decisions. The strict calibration failure remains recorded; no cutoff or model was changed. Both models reproduced every test classification. Fresh CPU scores provide additional calibration-fit evidence, but do not recover the missing original GPU raw exports. These checks establish engineering reproducibility within stated numerical limits, not clinical validity.')
    p('A separate actual-command smoke test (version 360927122) verified the frozen source, configuration and three checkpoint hashes, then invoked predict.py in six CPU subprocesses on the first six test images by ascending pixel hash. All commands completed; no classification changed and the maximum score difference was 0.000001205. This checks the user-facing entry point and supplements the full-cohort verification; it does not broaden the clinical evidence.')
    p('The work is not submitted to a journal. The source release destination is github.com/bekturovtemirlan000-gif/pneumonia-research-benchmark; source publication does not constitute peer review. Before submission, the responsible human authors must verify all claims and references, establish authorship and affiliations, determine applicable ethics requirements for their setting, disclose AI assistance and funding or conflicts accurately, and assess novelty and venue fit. No institutional approval, informed-consent exemption, author contribution or conflict declaration is invented here.')
    reference_start=len(story)
    h('References')
    references=[
      '[1] Kermany D, Zhang K, Goldbaum M. Labeled Optical Coherence Tomography and Chest X-Ray Images for Classification. Mendeley Data, version 2, 2018. https://doi.org/10.17632/rscbjbr9sj.2',
      '[2] Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. PLOS Medicine. 2018;15:e1002683. https://doi.org/10.1371/journal.pmed.1002683',
      '[3] Nazim-E-Alam. Cross-dataset transportability of pediatric chest X-ray deep learning across three countries. arXiv preprint, 2026. https://arxiv.org/abs/2609.05140. Peer-review status not independently verified.',
      '[4] Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. Proceedings of Machine Learning Research. 2017;70:1321-1330. https://proceedings.mlr.press/v70/guo17a.html',
      '[5] Durjoy SH and colleagues. BDCXR-3257 A Comprehensive Chest Radiography Dataset with Multi-Etiological Pneumonia Labels from a Clinical Setting. Figshare, 2026. https://doi.org/10.6084/m9.figshare.32086362',
      '[6] Checklist for Artificial Intelligence in Medical Imaging CLAIM 2024 Update. Radiology Artificial Intelligence. 2024;6:e240300. https://doi.org/10.1148/ryai.240300',
      '[7] Torchvision model documentation. Official implementation and pretrained-weight definitions. https://docs.pytorch.org/vision/master/models.html. Accessed 21 October 2026.',
      '[8] Wasimul Bari Rahat. BDCXR3257, Kaggle mirror version 1, including supporting documents. https://www.kaggle.com/datasets/wasimulbarirahat/bdcxr3257. Documentation inspection: Kaggle notebook version 360891626; source equivalence and permission scope unresolved.',
      '[9] Kermany DS, Goldbaum M, Cai W, et al. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018;172:1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010. Chest-radiograph population and labeling: STAR Methods, e1.'
    ]
    for reference in references: p(escape(reference),'Small')
    story[reference_start:]=[KeepTogether(story[reference_start:])]
    output.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Times-Roman',9); canvas.drawRightString(7.5*inch,.45*inch,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=(8.5*inch,11*inch),rightMargin=.9*inch,
        leftMargin=.9*inch,topMargin=.7*inch,bottomMargin=.7*inch,
        title='Architecture, resolution and calibration in pediatric pneumonia image classification',author='')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/pdf/pneumonia_research_report.pdf'))
    parser.add_argument('--benchmark',type=Path)
    args=parser.parse_args(); build(args.output,args.benchmark)
