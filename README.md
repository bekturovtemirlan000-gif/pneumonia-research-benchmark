# Pneumonia detection from chest X-rays

Completed exploratory PyTorch benchmark: three architectures, three seeds and two input resolutions (18 training runs). This is a binary image classifier (NORMAL / PNEUMONIA), not a clinical diagnostic system.

## Current selected model and results

Development-only selection across the six architecture/resolution candidates chose a **320-pixel EfficientNet-B0 three-seed ensemble**. All 18 trials finished with early stopping. The two training jobs took 134.58 minutes combined, including Kaggle overhead. Longer training is not evidence of better performance.

| Source-test measure | Selected 320-pixel ensemble | 95% group-bootstrap interval |
|---|---:|---:|
| AUROC | 0.96399 | 0.94700–0.97874 |
| Average precision | 0.96715 | 0.94498–0.98399 |
| Accuracy | 87.38% | 84.42–90.17% |
| Sensitivity | 99.74% | 99.16–100.00% |
| Specificity | 66.67% | 60.87–72.73% |
| Brier score | 0.13948 | 0.11123–0.16664 |

Temperature is 1.34055775 and the frozen calibration-selected cutoff is 0.83170023. Confusion counts: TN 154, FP 77, FN 1, TP 386. The test set contains 618 images in 426 filename-derived groups. Its earlier results were already inspected, so this is **exploratory**, not a fresh confirmatory evaluation.

The paired AUROC difference versus the development-selected 224-pixel ResNet18 ensemble is +0.00586 (95% interval −0.00546 to +0.01698). The interval does not establish improvement or equivalence. The two ensembles have identical aggregate confusion counts but different image-level predictions. The 320-pixel Brier score is worse than the 224-pixel value, despite its higher AUROC point estimate. Neither model was chosen using test metrics.

Both selected ensembles passed fresh CPU inference on all 618 test images with zero changed classifications; all 18 checkpoints loaded strictly. Maximum calibrated-score differences were 0.000004414 (224) and 0.000004643 (320), below tolerance 0.0001. This verifies engineering reproduction on the same cohort, not clinical or external validity.

[Frozen error analysis](https://www.kaggle.com/code/temirlanbekturov/pneumonia-frozen-error-analysis?scriptVersionId=360916945) found 531 images correct for both models, 69 incorrect for both, and nine correct only for each model. All 18 disagreements were NORMAL-labeled images. Both models missed the same one pneumonia-labeled image. Source-filename subgroup counts and their limitations are preserved in `results/frozen_error_analysis.json`; they do not establish clinical etiology or patient independence.

- [Research report](output/pdf/pneumonia_research_report.pdf): methods, all trials, uncertainty, provenance and limitations.
- [Cohort flow](output/figures/cohort_flow/cohort_flow.pdf): audited image counts and partition construction.
- [Model card](MODEL_CARD.md): intended use and model limitations.
- [Experiment registry](results/experiment_registry.json): immutable Kaggle version links and submitted source hashes.
- [Matched trial figure](output/figures/resolution_comparison/matched_resolution_trials.png): all 18 individual trials; lines are not confidence intervals.
- [320-pixel Kaggle run](https://www.kaggle.com/code/temirlanbekturov/pneumonia-research-resolution-320?scriptVersionId=360888746) and [CPU verification](https://www.kaggle.com/code/temirlanbekturov/pneumonia-resolution-320-cpu-verification?scriptVersionId=360906861). Kaggle access depends on notebook visibility and your account.

Compact results were captured from visible Kaggle logs and expanded output previews, with structural checks and independent in-Kaggle artifact validation. Complete new weight archives have not been downloaded locally and are not included here. The source notebooks can reproduce the experiments. External testing remains pending clarification of the candidate external dataset's release and access statements.

`python audit_research_results.py` checks consistency across compact trial, selection, comparison and verification records. It does not recompute image-level AUROC or bootstrap intervals. `results/research_record_audit.json` records its scope and outcome.

Original GPU raw calibration probabilities were not exported. A separate [CPU calibration reproduction](https://www.kaggle.com/code/temirlanbekturov/pneumonia-calibration-fit-reproduction?scriptVersionId=360912398) reran all 482 calibration images for each selected ensemble, exported fresh raw scores, and independently refit temperature and cutoff. Relative temperature differences were below 0.000002 and absolute cutoff differences below 0.0000014. The selected 320-pixel ensemble passed every check. The 224-pixel comparator failed the strict zero-changed-classification criterion on one calibration image, so the combined notebook correctly ended with a failed status after writing both reports.

[Boundary inspection](https://www.kaggle.com/code/temirlanbekturov/pneumonia-calibration-boundary-review?scriptVersionId=360914770) found that the comparator's saved score equaled its cutoff exactly; CPU inference moved it lower by 0.000001038. CSV round-trip precision did not change stored decisions. The original strict failure is retained, with no cutoff or model changes. Both ensembles still reproduced all 618 **test** classifications. Fresh CPU scores are additional evidence, not the missing original GPU raw exports. See `results/benchmark224_calibration_reproduction.json`, `results/benchmark320_calibration_reproduction.json` and the corresponding `calibration_boundary.json` records.

## Three-architecture research benchmark

`research_benchmark.py` and `pneumonia_research_benchmark.ipynb` define nine trials: EfficientNet-B0, DenseNet121 and ResNet18, each with seeds 42, 123 and 2026. [Kaggle version 360875315](https://www.kaggle.com/code/temirlanbekturov/pneumonia-research-benchmark-three-architectures?scriptVersionId=360875315) completed successfully in 3,048.6 seconds. Development selection chose the three-seed ResNet18 ensemble. Its logged test results are AUROC 0.95813, accuracy 87.38%, sensitivity 99.74% and specificity 66.67% (TN 154, FP 77, FN 1, TP 386). The exact logged result and frozen configuration are saved in `results/benchmark224_metrics.json` and `results/benchmark224_selection.json`; independent CPU checkpoint reproduction passed in Kaggle version 360888922. These results do not demonstrate uniform improvement over the earlier follow-up.

Temperature scaling fitted on the calibration partition worsened test Brier score from 0.12752 to 0.13105 and log loss from 0.97988 to 1.27754. It is therefore not described as successful test calibration. Score clipping before temperature scaling can introduce ties, so the reported raw and calibrated AUROCs can differ slightly. Neither the threshold nor the temperature was adjusted using this test result.

The audited training and test cohorts are frozen by decoded-pixel hash. The earlier validation cohort is divided by filename-derived patient group into 456 model-selection and 482 calibration images. Explicit frozen group membership avoids changing cohorts across scikit-learn versions. Architecture is chosen by mean selection AUROC across all three seeds; its seed probabilities are averaged. A positive temperature and Youden threshold are fit only on the separate calibration partition, then frozen before test evaluation. The entire original test remains exploratory because earlier results were inspected.

Training uses a 270-minute budget including auditing, with early stopping; queue time and final evaluation are separate. All trial histories and predictions are retained, including unsuccessful or time-limited outcomes. The script requires all nine trials before producing a final comparison. It uses one GPU even when Kaggle allocates T4 x2. Lossless resized-image caching reduces repeated decoding overhead, and evaluation uses float32 instead of training autocast.

`research_protocol.json` records the plan and submitted source hash. `external_evaluation_protocol.json` records prospective independent-data checks and frozen evaluation rules; external validation has not been completed. `predict.py` supports all three benchmark architectures, nested checkpoints and the exported temperature. `validate_artifacts.py` checks both the old three-part and new four-part manifests.

The prespecified resolution extension compares 320-pixel inputs against the initial 224-pixel arm, retaining the same cohorts, architectures and seeds. See `resolution_extension_protocol.json`; this extension is exploratory and does not replace the primary benchmark. `select_resolution.py` reconstructs the combined choice using only development predictions and checks that development cohorts agree. It never reads test predictions.

`inspect_selection_stability.py` describes how often each fixed architecture/resolution candidate wins when selection groups are resampled jointly. This exploratory addition was recorded after the 224-pixel results and before the 320-pixel test results in `selection_stability_protocol.json`. It retains the original full-sample choice. Winner frequencies are conditional stability measures, not probabilities of superiority; checkpoint selection used the same partition. `build_resolution_analysis.py` packages this analysis, the development-only choice and paired frozen-test uncertainty into `pneumonia_resolution_analysis.ipynb`.

Both arms completed; `plot_resolution_comparison.py` accepts their two `all_trial_results.json` files and requires all nine matched architecture/seed pairs in each. It exports PNG and vector PDF panels for development AUROC, exploratory test AUROC and calibrated test Brier score. Connecting lines identify matched individual trials, not confidence intervals or ensemble performance.

`build_verification_notebook.py` generates `pneumonia_export_verification.ipynb` from the exact local inference and verification sources. With the completed benchmark output and original image dataset attached, it checks artifact consistency, loads every checkpoint, and reproduces all selected-ensemble test predictions on CPU. The tolerance is 0.0001 in calibrated score with no changed binary predictions. This is an engineering reproducibility check, not external validation. The notebook also saves descriptive score/calibration plots and calibration-only threshold resampling, holding the fitted temperature fixed. These checks passed for the 224-pixel benchmark in Kaggle version 360888922: all nine checkpoints loaded, all 618 test-image identities matched, no classifications changed, and the maximum score difference was 0.000004414. Exact results are in `results/benchmark224_cpu_verification.json`.

The separate similarity-screening notebook completed. Its 64-bit perceptual-hash screen found 814 cross-partition candidate pairs, including 269 involving test images, among 5,371 retained images. These are not confirmed duplicates. Limited contact-sheet inspection found visible differences in the reviewed examples; it does not establish patient independence or exclude duplicates elsewhere. The counts and review limitations are recorded in `results/similarity_screening_summary.json`. No cohort exclusions or model changes were made from this screen.

Saved reliability-bin diagnostics show substantial differences between fitted calibration and source-test behavior. Ten-bin ECE is 0.00941 on calibration versus 0.16577 on test for the selected model; this is descriptive, bin-dependent, and has no uncertainty interval here. All bin counts are retained in `results/benchmark320_prediction_diagnostics.json` and the corresponding 224-pixel record. `plot_calibration_diagnostics.py` creates the report's reliability figure. These records distinguish the diagnostic log-loss clipping rule from the main benchmark rule: the 224-pixel test log losses are 1.16079 and 1.27754 respectively. Neither convention was changed after evaluation.

## Reproduce the current research comparison on Kaggle

1. Import `pneumonia_research_benchmark.ipynb` into a new Kaggle notebook. Attach [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), select a GPU and enable Internet for official ImageNet weights. Use **Save Version → Save & Run All**. This runs nine 224-pixel trials and writes `/kaggle/working/pneumonia_benchmark_outputs`.
2. After completion, import `pneumonia_resolution320.ipynb` into a separate notebook with the same dataset and GPU settings. Save and run it. This runs nine 320-pixel trials and writes `/kaggle/working/pneumonia_resolution320_outputs`. Run the arms sequentially to match the recorded workflow. Different hardware or packages can change results.
3. Import `pneumonia_export_verification.ipynb` into a CPU notebook. Attach the original dataset and your completed 224-pixel notebook output. Save and run it. Repeat with `pneumonia_export_verification_320.ipynb`, attaching your 320-pixel output and the original dataset. These notebooks verify checkpoints, pixel identities, metrics and all selected-ensemble test predictions.
4. Import `pneumonia_resolution_analysis.ipynb` into a CPU notebook and attach **both** completed training outputs. It saves the development-only choice before reading test predictions, then reports selection stability and paired test uncertainty. It requires matching cohorts.
5. For calibration-fit reproduction, import `pneumonia_calibration_reproduction.ipynb` into a CPU notebook with **both** completed training outputs and the original image dataset. It checks all 482 calibration images per model and exports fresh raw CPU scores. A strict failure can occur after both reports are saved; read each report separately. To inspect score crossings without changing the criteria, run `pneumonia_calibration_boundary.ipynb` with that calibration notebook's output attached.
6. For descriptive error overlap and filename subgroups, run `pneumonia_frozen_error_analysis.ipynb` on CPU with both completed training outputs. This reads frozen test predictions; it does not choose models or fit thresholds. Filename subgroup annotations are not independently verified clinical diagnoses.
7. For the post hoc dimension-only control, run `pneumonia_geometry_control.ipynb` on CPU with the completed 224-pixel output and original image dataset. It verifies all retained pixel identities, then fits a fixed logistic regression using log width and log height on training images only. Image loading can take substantially longer than fitting. This control is separate from the selected neural model.
8. Run `pneumonia_geometry_analysis.ipynb` on CPU with the completed geometry-control output and completed 320-pixel training output. It reconstructs every geometry score from the saved coefficients, verifies partition metrics, and computes paired uncertainty with direction geometry minus neural.
9. To exercise the actual single-image command, run `pneumonia_inference_cli_smoke.ipynb` on CPU with the **original recorded** 320-pixel output (version 360888746) and original dataset. It checks the frozen configuration, inference source and checkpoint hashes, then calls `predict.py` in six separate processes on the first six test hashes. This supplements the full-cohort check; it is not external validation. Its embedded lock intentionally rejects newly retrained weights. To verify your independent rerun, create a separate documented model lock from that run and rebuild the notebook; do not label it verification of the original weights.
10. Retain immutable Kaggle version IDs and download complete outputs when possible. Keep `inference_config.json` beside its checkpoint subdirectories. GitHub contains compact evidence records and training source; it does not contain the new checkpoint archives or per-image predictions.

Training requires the original audited source cohort and deliberately fails if cohort digests differ. Do not disable those checks and treat a changed dataset as the same experiment. A different dataset requires a separate documented protocol.

The recorded training packages are in `results/benchmark224_environment.json` and `results/benchmark320_environment.json`. `requirements-local-lock.txt` describes separate local development, not the Kaggle container. Original saved outputs are authoritative. Registry links may require access to the owner's private Kaggle notebooks; the public source notebooks allow independent reruns under your account.

`results/external_model_lock.json` identifies the selected model, frozen preprocessing and decision rule, inference source hash, and three checkpoint hashes for any future external evaluation. It does not authorize dataset access or claim that external testing was performed. Source-access issues remain documented in `external_evaluation_protocol.json`.

Local checks without training:

```powershell
python audit_reproducibility.py
python audit_research_results.py
python -m unittest discover -s tests -v
```

The first compares embedded code with local modules and submitted hashes. The second cross-checks compact records without recreating per-image metrics. Tests exercise corrupted artifacts, cohort mismatch and selection independence. These local checks do not attest remote execution.

For historical experiments, `pneumonia_kaggle.ipynb` produces `pneumonia_outputs`, and `pneumonia_ensemble_kaggle.ipynb` produces `pneumonia_ensemble_outputs`. Their shared validation design differs from the current four-partition comparison. Do not substitute those outputs into current benchmark analysis.

## Historical baseline experiment

The provided test partition is held out. Original train and validation images form a development pool. Filename-derived patient groups define a new approximately 80/20 development split. Exact decoded-pixel duplicates are removed, and development groups overlapping test are excluded. The run fails on unrecognized filename patterns. These IDs are proxies: no claim of verified patient independence or near-duplicate exclusion is made.

Full-image letterboxing to 224×224, small affine training augmentation, ImageNet normalization, weighted binary cross entropy, AdamW, validation-AUROC checkpoint selection and early stopping. The decision threshold maximizes validation Youden J and is frozen before test evaluation. One seed (42) establishes a baseline; GPU results can still vary across environments.

Saved artifacts include model weights, a split manifest, audit counts, per-image predictions, training history, environment versions, test ROC/PR/confusion plots and metrics with subject-proxy cluster bootstrap intervals. Average precision is reported explicitly, not mislabeled as trapezoidal PR AUC. Confidence intervals describe test sampling uncertainty, not training-seed uncertainty.

## Historical baseline and follow-up

The baseline and three-seed follow-up completed successfully. Validation selected the seed-123 single model over the averaged ensemble. At its validation-selected threshold, test accuracy was 88.19%, sensitivity 99.22%, specificity 69.70%, and AUROC 0.96844. Accuracy and specificity increased relative to the baseline, while sensitivity, ranking metrics and calibration did not improve. See `MODEL_CARD.md` for the comparison and limitations. The follow-up is exploratory because the original test results had already been inspected. A single dataset does not establish a new scientific contribution. The current report adds comparator models and a documented question; independent external validation and subgroup/error analysis remain limited by source access and metadata. Dataset-specific performance does not establish clinical safety or generalization to other hospitals or populations.

## Exported model use

`predict.py` loads the selected checkpoints from `inference_config.json` without downloading weights or starting training. For baseline exports, create that configuration from the recorded threshold and `best_weights.pt`; the downloaded local baseline already includes it.

```powershell
python predict.py --model-dir artifacts/baseline_v1/pneumonia_outputs --image path/to/xray.jpeg
python predict.py --model-dir artifacts/ensemble_v3/pneumonia_ensemble_outputs --image path/to/xray.jpeg
python validate_artifacts.py artifacts/baseline_v1/pneumonia_outputs
python -m unittest discover -s tests -v
```

For the **current selected 320-pixel ensemble**, download your completed resolution notebook output, then use its `pneumonia_resolution320_outputs` directory:

```powershell
python predict.py --model-dir path/to/pneumonia_resolution320_outputs --image path/to/xray.jpeg --device cpu
```

Keep `inference_config.json` and the three `efficientnet_b0_seed_*/best_weights.pt` subdirectories together. The command loads the model locally, averages the three sigmoid scores, applies the saved temperature, and uses the saved threshold. It does not train or download model weights. The examples above reference historical experiments, whose checkpoints are not interchangeable with the selected model.

Prediction requires PyTorch, torchvision and Pillow. The output score is not a calibrated clinical probability. The model does not verify that an input is a valid chest X-ray. Do not interpret predictions on arbitrary images as medical findings.

## Follow-up experiment

`train_ensemble.py` and `pneumonia_ensemble_kaggle.ipynb` train up to three independently initialized EfficientNet-B0 members (seeds 42, 123, 2026), using lower learning rates and validation-driven learning-rate decay. The data split is fixed. DataLoader sampling/augmentation seeds are shared in this version, so the runs vary model initialization and stochastic model behavior but are not fully independent data-order experiments. Validation AUROC selects between individual members and an equal-weight probability ensemble. It then selects a threshold before test evaluation. The 110-minute development budget includes dataset auditing, with approximately 10 minutes reserved for export/evaluation; queue and container startup delays are outside that budget. Early stopping can end members earlier.

Files under `artifacts/`, model weights and browser screenshots are excluded from Git. Keep the original downloaded outputs and run IDs for provenance.

`results/experiment_registry.json` maps experiments to immutable Kaggle version IDs. `results/benchmark224_environment.json` records the actual GPU training packages; `requirements-local-lock.txt` describes the separate local CPU environment. `python package_research.py` builds `artifacts/research_source_bundle.zip` from an explicit code/results/report allowlist, checks notebook source consistency and verifies every archived file against its SHA-256 checksum. The package includes a checksum manifest. Its status label describes the release preparation state, not peer review or clinical readiness. Rebuild it after changing results or the PDF.

`results/` contains the measured baseline and follow-up metrics, model-selection record, paired comparison, data-audit counts and training environment. These compact records are suitable for source control; raw images and model weights are not included.

To reproduce the historical follow-up comparison, validate both downloaded exports before comparing them:

```powershell
python validate_artifacts.py path/to/ensemble_outputs
python compare_experiments.py artifacts/baseline_v1/pneumonia_outputs path/to/ensemble_outputs --output artifacts/paired_comparison.json
```

The comparison pairs identical image hashes and filename-derived patient groups, retains each experiment's frozen predictions, and reports paired cluster-bootstrap intervals for metric differences. It rejects different test cohorts. Results remain exploratory; these intervals do not capture training or model-selection uncertainty.

## Sources and attribution

- Kermany, Zhang and Goldbaum. *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*, version 2 (2018). [DOI: 10.17632/rscbjbr9sj.2](https://data.mendeley.com/datasets/rscbjbr9sj/2). Source lists CC BY 4.0; verify the attached Kaggle version's terms before redistributing data. No images are included in this repository.
- [Kaggle dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
- [Torchvision EfficientNet-B0 documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html). The experiment uses full-image letterboxing instead of the pretrained weights' standard center crop.


## Post hoc image-dimension control

A fixed logistic regression using only log source-image width and height achieved test AUROC **0.9084** (95% filename-group bootstrap interval 0.8807–0.9343), accuracy **80.91%**, sensitivity **96.38%**, specificity **54.98%** and Brier score **0.13329**. Confusion counts: TN 127, FP 104, FN 14, TP 373. The scaler and classifier were fit only on 3,815 training images; threshold 0.5 was fixed, with no hyperparameter search or calibration. Pixel hashes verified identity and were not model features.

The neural ensemble's AUROC exceeded this control by 0.0556. The archived paired analysis uses **geometry minus neural**, giving −0.0556 (95% interval −0.0798 to −0.0325). The Brier difference was −0.0062 (−0.0223 to 0.0106), so an overall calibration advantage is not established. These are exploratory, conditional intervals from 1,000 resamples of 426 filename-derived groups; they omit training and selection uncertainty.

The control was specified after neural test results were inspected and before its own results were observed. Strong predictability from dimensions signals a dataset association; it does not prove that the neural network relies on dimensions or establish a causal explanation. Dimensions are measured in pixels, not physical units. No checkpoint, cohort or cutoff was changed.

[Control run](https://www.kaggle.com/code/temirlanbekturov/pneumonia-image-geometry-control?scriptVersionId=360920257) completed in 1,221.4 seconds. [Independent coefficient reconstruction and comparison](https://www.kaggle.com/code/temirlanbekturov/pneumonia-geometry-control-verification?scriptVersionId=360924812) completed in 33.5 seconds: all 5,371 scores reconstructed exactly, with zero changed classifications. Full records are in `results/geometry_control.json` and `results/geometry_analysis.json`.


The [actual inference-command smoke test](https://www.kaggle.com/code/temirlanbekturov/pneumonia-inference-command-verification?scriptVersionId=360927122) also passed. It verified the frozen source, configuration and all three checkpoint hashes, then ran `predict.py` in six separate CPU subprocesses on the first six test images by ascending pixel hash. All commands exited successfully, no classification changed, and the maximum score difference was 0.000001205. This small entry-point check supplements the separate 618-image verification; it is not external validation. See `results/inference_cli_smoke.json`.

The paper now distinguishes source-reported population and labeling procedures from this project's own checks. See `results/source_population_review.json` and the model card; no independent diagnosis adjudication or demographic audit is claimed.

`results/reporting_gap_review.json` records a focused review against selected CLAIM 2024 topics, including unavailable acquisition/demographic information and author declarations still needed before submission. It is not a completed checklist, compliance score or peer review.

`results/seed_variation_summary.json` and Supplementary Table S1 describe mean and sample standard deviation across the three individual models in each architecture/resolution group. They are post hoc summaries, not confidence intervals or ensemble performance. The frozen model choice is unchanged. Recreate the summary with `python summarize_seed_variation.py`.
