# Pneumonia classifier: research model card

## Intended use

Educational and research study of binary chest X-ray classification. Not validated for clinical decisions, triage, screening, diagnosis, or use on arbitrary uploaded images. A label of NORMAL means the dataset class, not an assurance that the person is healthy.

## Current development-selected model

EfficientNet-B0, 320-pixel RGB letterboxed input, three seeds (42, 123, 2026). Average member sigmoid probabilities, clip to [1e-7, 1-1e-7], transform to logit, divide by temperature **1.340557747633612**, then apply sigmoid. Predict PNEUMONIA at scores at least **0.8317002265352794**. Architecture and resolution were selected by mean development AUROC; temperature and cutoff were fit on a separate calibration partition before test evaluation.

Training/selection/calibration/test contain 3,815/456/482/618 images. Selection uses 270 filename groups and calibration uses 268. The same held-out images were used by earlier experiments, so the reported test evaluation is exploratory.

| Test metric | Estimate | Descriptive 95% group-bootstrap interval |
|---|---:|---:|
| AUROC | 0.96399 | 0.94700–0.97874 |
| Average precision | 0.96715 | 0.94498–0.98399 |
| Accuracy | 87.38% | 84.42–90.17% |
| Sensitivity | 99.74% | 99.16–100.00% |
| Specificity | 66.67% | 60.87–72.73% |
| Brier score | 0.13948 | 0.11123–0.16664 |

TN 154, FP 77, FN 1, TP 386. The false-positive rate among the 231 NORMAL-labeled test images is 33.33%. This does not support clinical deployment. Intervals use 1,000 resamples of 426 filename groups and omit training, model-selection and external-population uncertainty.

Compared with the 224-pixel ResNet18 ensemble, AUROC differs by +0.00586 (paired interval −0.00546 to +0.01698) and Brier score by +0.00843 (−0.00125 to +0.01832). Neither improvement nor equivalence is established. Identical confusion totals do not imply identical predictions. Within the 320-pixel ensemble, temperature scaling reduced test Brier score from 0.14313 to 0.13948 and log loss from 1.15973 to 0.78876; the 224-pixel ensemble showed the opposite calibration effect.

All nine 320-pixel checkpoints loaded and ran on CPU. Fresh selected-ensemble inference on all 618 hash-matched test images changed zero classifications; maximum score difference was 0.000004643. [Verification run 360906861](https://www.kaggle.com/code/temirlanbekturov/pneumonia-resolution-320-cpu-verification?scriptVersionId=360906861) passed in 433.3 seconds. Exact results and checkpoint hashes are in `results/benchmark320_cpu_verification.json`.

Selection-group resampling retained this candidate in 852/1,000 replicates. That frequency is descriptive stability, not a probability of superiority. Calibration-only cutoff percentiles were 0.60672–0.89223 (2.5th–97.5th), holding temperature fixed; the cutoff was not changed. Full results for all 18 trials are retained, including models with higher test scores that were not selected.

Original GPU raw calibration probabilities were not exported. Fresh CPU inference on all 482 calibration images now exports raw scores and independently refits temperature and cutoff. The selected 320-pixel ensemble passed: relative temperature difference 0.000001736, absolute cutoff difference 0.000000940, and zero changed calibration classifications. The 224-pixel comparator had one changed calibration classification: its saved score exactly equaled the cutoff and CPU inference moved it lower by 0.000001038. Its strict zero-change check remains failed; neither model nor cutoff was altered. Both models reproduced all test classifications. These fresh CPU exports do not replace the missing original GPU raw scores. Kaggle versions 360912398 (both calibration fits; combined strict failure) and 360914770 (successful boundary inspection) preserve this evidence.

Frozen error analysis found 531 test images correct for both ensembles, 69 incorrect for both, and nine correct only for each model. All 18 disagreements were NORMAL-labeled images; both models missed the same one pneumonia-labeled image. Both detected 239/240 bacterial-filename and 147/147 viral-filename pneumonia images. These post hoc counts describe binary classification of source labels, not differential diagnosis or guaranteed future sensitivity. One filename proxy group spans both pneumonia subgroups, so subgroup group counts are not independent patient counts. See `results/frozen_error_analysis.json` and Kaggle version 360916945.

The selected model's ten-bin ECE is 0.00941 on its fitted calibration partition and 0.16577 on source test. Mean score for NORMAL-labeled images is 0.02119 versus 0.44694. These descriptive diagnostics expose poor transport of development calibration; they do not identify its cause or validate patient-level probabilities. Sparse-bin counts are retained with the figure. Diagnostic log loss uses extra clipping to [1e-7, 1-1e-7], so its values must not be mixed with the main benchmark's default-clipping log loss.

The following sections preserve historical baseline and comparison results; their configurations are distinct from the current selected model.

## Completed baseline

- Kaggle version: [360813730](https://www.kaggle.com/code/temirlanbekturov/pneumonia-detection-efficientnet-baseline?scriptVersionId=360813730).
- Architecture: EfficientNet-B0, ImageNet initialization, one binary output.
- Training: 22 epochs executed; early stopping; checkpoint from epoch 14 selected by validation AUROC.
- Runtime: Kaggle reports 42 minutes 51 seconds overall. The saved training/evaluation timer reports 1,725.68 seconds and excludes the initial dataset audit.
- Split: 3,815 training images, 938 validation images, 618 test images after exclusions.
- Audit: 5,856 initial images; 479 development images excluded through overlap/duplicate rules; 6 duplicate test images removed.
- Threshold: 0.0757213681936264, chosen by maximum validation Youden J.

| Metric | Validation | Test |
|---|---:|---:|
| AUROC | 0.99894 | 0.97061 |
| Average precision | 0.99957 | 0.97686 |
| Accuracy | 98.83% | 80.42% |
| Balanced accuracy | 98.40% | 73.81% |
| Sensitivity | 99.40% | 100.00% |
| Specificity | 97.41% | 47.62% |
| F1 | 0.99178 | 0.86480 |

Test confusion matrix: 110 true negatives, 121 false positives, 0 false negatives, 387 true positives. At the separately reported fixed threshold of 0.5, test accuracy was 84.47% and specificity 58.87%; this does not justify switching the deployed threshold after examining test results.

The score ranks images reasonably well in this test sample, but the selected decision rule generalizes poorly to normal test images. No perfect-performance claim is justified. Sensitivity of 100% in this finite test sample does not imply perfect sensitivity in future patients. A percentile bootstrap interval of [1,1] occurs because every sampled positive was classified correctly; it does not express all uncertainty about future populations.

## Verification

The downloaded predictions reproduce the recorded accuracy, sensitivity and specificity. The saved manifest has no overlapping exact pixel hashes or filename-derived patient groups across partitions. This check does not establish independently verified patient identity, eliminate near-duplicates, or rule out acquisition/source confounding.

The test partition contains 426 filename-derived groups (224 NORMAL and 202 PNEUMONIA), with up to nine test images per group. The development partitions contain 2,153 training groups and 538 validation groups. Reported metrics are image-level; bootstrap resampling preserves within-group image clustering. The executed baseline notebook source and the submitted follow-up source were each checked against their local Python files and matched after line-ending normalization.

Normal-image mean scores were 0.01834 in validation and 0.40409 in test. This is a descriptive distribution difference, not proof of its cause. The dataset lacks the information needed here for a complete demographic or hospital-specific fairness/generalization audit.

Descriptive calibration analysis found a Brier score of 0.01527 on validation versus 0.13507 on test (lower is better). The 10-bin equal-width expected calibration error was 0.01463 versus 0.14713. These bin-dependent estimates include sparsely populated bins and should not be treated as precise clinical calibration measurements. `analyze_predictions.py` saves bin counts alongside the plot.

A separate validation-only stability analysis resampled 538 filename-derived patient groups 1,000 times and recomputed maximum-Youden thresholds. The threshold distribution had median 0.10017, interquartile range 0.07572–0.56028, and 2.5th–97.5th percentiles 0.04361–0.98781. This suggests sensitivity of the selected cutoff to the development sample. These descriptive percentiles are not a guaranteed interval for an optimal clinical cutoff, and the same validation set had already been used for checkpoint selection. The saved deployment threshold was not changed. Run `inspect_threshold_stability.py` to reproduce the calculation without accessing test predictions.

The exported state dictionaries were checked for exact equality, and the prediction CLI successfully loaded the baseline on CPU and produced deterministic finite scores on a synthetic fixture. This is an engineering smoke test, not medical validation. Local testing used a newer PyTorch release than the Kaggle training container; versions are recorded separately in `requirements-local-lock.txt` and the exported `environment.json`.

A real-image reproducibility check used `IM-0001-0001.jpeg`. Its decoded-pixel hash matched the saved test manifest. CPU float32 inference scored 0.10366851 versus the saved Kaggle GPU-autocast score of 0.10762492 (absolute difference 0.00395641). Both classified this NORMAL example as PNEUMONIA at the frozen threshold. This one-image check confirms compatible loading and preprocessing, but not exact cross-environment reproducibility; the effects of numeric precision and library versions have not been isolated.

## Follow-up and research limitations

Version 2 (three seeds and learning-rate decay) was canceled with no output. [Version 3](https://www.kaggle.com/code/temirlanbekturov/pneumonia-detection-efficientnet-baseline?scriptVersionId=360829588) completed successfully. Seeds 42, 123 and 2026 ran 22, 20 and 22 epochs, respectively. Their selected checkpoints were epochs 14, 12 and 14. Validation AUROC selected seed 123 (0.99906576), ahead of the equal-weight ensemble (0.99903526). The exported predictor therefore uses one checkpoint, not all three. Its threshold, selected on validation, is 0.9298236966133118.

| Test metric | Baseline | Selected follow-up |
|---|---:|---:|
| AUROC | 0.97061 | 0.96844 |
| Average precision | 0.97686 | 0.97306 |
| Accuracy | 80.42% | 88.19% |
| Sensitivity | 100.00% | 99.22% |
| Specificity | 47.62% | 69.70% |
| Brier score (lower is better) | 0.13507 | 0.14308 |

Follow-up confusion matrix: 161 true negatives, 70 false positives, 3 false negatives, 384 true positives. Validation-selected decision rules improved observed accuracy and specificity while losing sensitivity. Ranking and calibration did not improve: observed AUROC and average precision decreased, and Brier score increased. At the common fixed threshold of 0.5, follow-up accuracy was 84.30% versus baseline 84.47%. The gain under the separately selected thresholds should not be described as an across-the-board improvement in the classifier.

Paired resampling of the same 426 test groups (1,000 replicates) gave an accuracy difference of +7.77 percentage points, with a descriptive 95% percentile interval of +5.53 to +10.27 points. The AUROC difference was -0.00217, with interval -0.01351 to +0.00802. These intervals describe test-sample variability, not model-selection or training uncertainty, and no multiple-comparison adjustment was applied. Both artifact sets passed split-identity and metric consistency checks. The exported follow-up checkpoint loaded and ran on CPU on a hash-verified public test image.

All follow-up comparisons on the original test partition are exploratory because the baseline test performance had been inspected. A fresh external dataset, with compatible label definitions and independently verified provenance, is needed for a confirmatory evaluation.

No research manuscript should present this baseline as a novel algorithm. A defensible project could instead investigate how patient grouping, duplicate removal, validation construction, and threshold transfer affect reported results. That study needs prespecified comparisons, repeatable splits, appropriate uncertainty estimates and external validation.

## Completed 224-pixel architecture benchmark

[Kaggle version 360875315](https://www.kaggle.com/code/temirlanbekturov/pneumonia-research-benchmark-three-architectures?scriptVersionId=360875315) trained EfficientNet-B0, DenseNet121 and ResNet18 with three seeds each. The original training/test cohorts stayed fixed; the previous validation cohort became 456 model-selection and 482 calibration images, separated by filename-derived group. Mean selection AUROC selected the ResNet18 ensemble. Its three probabilities are averaged, clipped to [1e-7, 1-1e-7], and temperature-scaled with temperature 0.67293227. The frozen cutoff is 0.72090517.

| Source test metric | Estimate | Descriptive 95% group-bootstrap interval |
|---|---:|---:|
| AUROC | 0.95813 | 0.93952–0.97529 |
| Accuracy | 87.38% | 84.47–90.32% |
| Sensitivity | 99.74% | 99.16–100.00% |
| Specificity | 66.67% | 60.37–73.02% |
| Brier score | 0.13105 | 0.10311–0.15994 |

Confusion counts are TN 154, FP 77, FN 1 and TP 386. Relative to the earlier selected follow-up, accuracy differs by -0.81 percentage points (paired interval -2.56 to +0.84) and AUROC by -0.01031 (interval -0.02579 to +0.00527). These intervals do not establish improvement or equivalence. The Brier difference is -0.01203 (interval -0.02421 to -0.00062), an exploratory comparison without multiplicity adjustment. All intervals use 1,000 resamples of the same 426 filename groups and omit training and selection uncertainty.

For this ensemble, temperature scaling worsened test Brier score from 0.12752 to 0.13105 and log loss from 0.97988 to 1.27754. Calibration-only threshold resampling gave 2.5th–97.5th percentiles 0.18496–0.94266, holding temperature fixed. These observations do not justify retuning on the source test.

[CPU verification version 360888922](https://www.kaggle.com/code/temirlanbekturov/pneumonia-checkpoint-cpu-verification?scriptVersionId=360888922) reconstructed the ensemble and all reported trial metrics, loaded all nine checkpoints, verified the decoded pixels of all 618 source test images and reproduced every selected-model classification. Maximum score difference was 0.000004414. This verifies engineering reproduction, not medical validity.

The person-number grouping rule may merge unrelated bacterial and viral subjects whose filename numbers coincide. Exact-hash separation does not establish true patient independence. A perceptual-hash screen flagged 814 cross-partition candidate pairs, 269 involving test; they are not confirmed duplicates, and most remain unreviewed. The completed 320-pixel extension is described above. Independent external inference remains pending clarification of conflicting BDCXR access statements.

## Data attribution

[Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), with original source [Kermany, Zhang and Goldbaum, Mendeley Data v2](https://doi.org/10.17632/rscbjbr9sj.2). Do not redistribute images without verifying applicable licensing and attribution requirements.


## Post hoc image-dimension control

A fixed logistic regression using only log source-image width and height achieved test AUROC **0.9084** (95% filename-group bootstrap interval 0.8807–0.9343), accuracy **80.91%**, sensitivity **96.38%**, specificity **54.98%** and Brier score **0.13329**. Confusion counts: TN 127, FP 104, FN 14, TP 373. The scaler and classifier were fit only on 3,815 training images; threshold 0.5 was fixed, with no hyperparameter search or calibration. Pixel hashes verified identity and were not model features.

The neural ensemble's AUROC exceeded this control by 0.0556. The archived paired analysis uses **geometry minus neural**, giving −0.0556 (95% interval −0.0798 to −0.0325). The Brier difference was −0.0062 (−0.0223 to 0.0106), so an overall calibration advantage is not established. These are exploratory, conditional intervals from 1,000 resamples of 426 filename-derived groups; they omit training and selection uncertainty.

The control was specified after neural test results were inspected and before its own results were observed. Strong predictability from dimensions signals a dataset association; it does not prove that the neural network relies on dimensions or establish a causal explanation. Dimensions are measured in pixels, not physical units. No checkpoint, cohort or cutoff was changed.

[Control run](https://www.kaggle.com/code/temirlanbekturov/pneumonia-image-geometry-control?scriptVersionId=360920257) completed in 1,221.4 seconds. [Independent coefficient reconstruction and comparison](https://www.kaggle.com/code/temirlanbekturov/pneumonia-geometry-control-verification?scriptVersionId=360924812) completed in 33.5 seconds: all 5,371 scores reconstructed exactly, with zero changed classifications. Full records are in `results/geometry_control.json` and `results/geometry_analysis.json`.


The [actual inference-command smoke test](https://www.kaggle.com/code/temirlanbekturov/pneumonia-inference-command-verification?scriptVersionId=360927122) also passed. It verified the frozen source, configuration and all three checkpoint hashes, then ran `predict.py` in six separate CPU subprocesses on the first six test images by ascending pixel hash. All commands exited successfully, no classification changed, and the maximum score difference was 0.000001205. This small entry-point check supplements the separate 618-image verification; it is not external validation. See `results/inference_cli_smoke.json`.
