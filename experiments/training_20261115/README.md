# November 2026 training and verification

The primary experiment completed nine fresh models. Development accuracy selected the three-seed regularized EfficientNetV2-S ensemble, but its frozen decision rule achieved **83.98% source-test accuracy**, compared with **87.38%** for the original benchmark comparator. No threshold was changed after observing that result. Full CPU reproduction passed on all 1,100 calibration/test images with zero changed classifications. Secondary grouped training completed 27 additional models; its independent CPU verification passed.

The grouped assessment produced one out-of-fold prediction per original training image: standard B0 was correct on 3,764/3,815 (98.66%), regularized B0 on 3,772 (98.87%), and regularized V2-S on 3,774 (98.93%). All checkpoints were fixed before outer scoring. These internal scores do not measure source-test or external clinical performance and do not change the primary model. The 27 fits completed 481 epochs in a 158.13-minute training region; 23 used early stopping, two reached a time allowance and two reached the 36-epoch limit. Two incomplete epochs were retained in the history and were ineligible for checkpoint selection.

Independent CPU reconstruction **passed**: all 27 checkpoint hashes and strict loads, all 3,815 image identities and all nine outer-fold ensembles were checked. Fresh inference changed 0 classifications; the largest score difference was 5.60283661e-06 (required tolerance 0.0001 and zero changed classifications). All four complete out-of-fold CSV exports matched their Kaggle byte hashes; a separate local audit reproduced every stored metric and decision. Two exported cutoff fields differ from their original JSON values by one floating-point unit (-5.55e-17) because the GPU CSV was parsed before export. The audit reproduces that parser path exactly and checks every saved and fresh decision against the authoritative JSON cutoff; none changes. The initial exact-equality audit failure and its correction are documented in `cv_threshold_serialization_review.json`; neither the exports nor the model settings were altered.

The prespecified dimension-only logistic-regression control scored 3,225/3,815 correct (84.53%), with AUROC 0.91874. It used log width and height, the same inner/outer groups, fixed hyperparameters and inner-only cutoff fitting. Its three configurations were frozen before outer scoring, and exported coefficients reproduced the probabilities within 1e-12. This establishes an association in this dataset, not that the neural networks exploit image dimensions.

This folder contains exploratory research, not clinical validation. All original source cohorts had already been used in earlier experiments. Filename groups are patient proxies, and independent external testing is incomplete.

## Protocol and provenance

| Experiment | Frozen source | Plan | Execution record |
|---|---|---|---|
| Nine primary fits | `submitted_source.py` | `protocol.json` | `run_record.json` |
| Three-fold, three-seed, three-recipe internal assessment | `submitted_grouped_cv.py` | `cv_protocol.json` | `cv_run_record.json` |
| Primary full-cohort CPU reproduction | `submitted_verification.py` | `verification_plan_record.json` | `cpu_run_record.json` |
| Grouped-CV full-cohort CPU reproduction and dimension control | `submitted_cv_verification.py` | `verification_plan_record.json`, `cv_geometry_protocol.json` | `cv_cpu_run_record.json` |
| Original comparator score export (retrospective, no fitting) | `submitted_comparator_export.py` | `previous_lock.json` | `comparator_export_run_record.json` |

Each submitted file is also embedded in its corresponding `.ipynb`. Source hashes normalize CRLF to LF, matching the verified full Kaggle editor clipboard. Checkpoint and exported-file digests concern exact binary bytes. GitHub commit identifiers in the run records establish the prospective versions.

The repository's `.gitattributes` disables automatic line-ending conversion so cloned source and data files retain their recorded bytes. This matters for exact CSV and manifest hashes on Windows as well as other platforms; see the [Git attributes documentation](https://git-scm.com/docs/gitattributes). Preserve the exported CSV bytes when auditing them rather than resaving them through a spreadsheet editor.

The original single-seed CV proposal was amended to three seeds before submission and never executed. The first 27-fit commit remained queued and was cancelled after zero execution seconds; `cv_run_record_v1_cancelled.json` preserves that attempt. The retry uses the identical training source. Neither the retry nor the separate CPU checks changes the primary model.

## Reproduce in Kaggle

1. Import `pneumonia_training.ipynb`, enable a GPU and Internet, and attach the original Chest X-Ray Images (Pneumonia) dataset and the original completed 320-pixel benchmark output. The code validates input cohort identities, configuration and checkpoint hashes before fitting.
2. Use Save & Run All. Training fits nine models, selects a development ensemble, freezes its calibration-only cutoff and checkpoints, then scores the previously inspected source test. The 225-minute training cap excludes auditing and final evaluation; early stopping can use much less time.
3. Attach that completed output and the original dataset to `pneumonia_cpu_verification.ipynb`, use CPU, and run it. The packaged verification is pinned to the observed configuration hash. A new training rerun requires rebuilding the verifier for its newly observed frozen hash; the existing execution record must remain historical.
4. The separate `pneumonia_grouped_cv.ipynb` uses only the original training cohort, trains 27 fresh models, and produces out-of-fold scores. Its three recipe ensembles use inner-only checkpoint and cutoff selection. It does not select a replacement primary model or score the original selection, calibration or test images.
5. Attach that completed grouped output and the original dataset to `pneumonia_cv_cpu_verification.ipynb`, use CPU, and run it. This verifier is pinned to the observed completed summary hash, checks all 27 checkpoints, repeats inference on every outer image, runs the prespecified dimension-only comparator, and exports four complete prediction tables. A new training rerun needs a newly pinned verifier; retain the historical execution records.

Required inputs are discovered beneath `/kaggle/input`; ambiguous or mismatched artifacts cause an error. Complete weight archives remain in the Kaggle outputs, subject to notebook visibility. They are not included in this repository.

The observed primary training run is [Kaggle version 369624654](https://www.kaggle.com/code/temirlanbekturov/pneumonia-regularized-training-november-2026?scriptVersionId=369624654); its independent CPU check is [version 369643988](https://www.kaggle.com/code/temirlanbekturov/pneumonia-november-cpu-reproduction?scriptVersionId=369643988). The grouped assessment uses [version 369645087](https://www.kaggle.com/code/temirlanbekturov/pneumonia-grouped-training-cross-validation?scriptVersionId=369645087). A link does not imply that a private Kaggle output is accessible to every reader.

The observed Kaggle environment uses scikit-learn **1.6.1**. A local preflight with **1.9.1** produced different `StratifiedGroupKFold` assignments despite the same seed; that preflight validates mechanics, not the actual remote fold membership. The captured Kaggle fold records and exported out-of-fold `fold` column are authoritative for this execution. Exported fold IDs are 0, 1 and 2; report labels are 1, 2 and 3 in the same order. Reproduce its assignments with the recorded environment, rather than assuming that a current library release has identical behavior. Recorded core versions are in `environment.json` and `cv_environment.json`. These are partial environment snapshots: the training run did not log Pillow's version or a container-image digest. The root `requirements-local-lock.txt` describes local analysis software, not the Kaggle training environment.

## Run the exported primary model

After downloading the completed primary output, keep `inference_config.json` and its referenced member folders together. Install the recorded PyTorch and torchvision versions with their compatible Pillow dependency, then classify a research image using:

```sh
python experiments/training_20261115/model.py --model-dir /path/to/pneumonia_november_outputs --image /path/to/image.jpeg --device cpu
```

The model directory is the folder containing `inference_config.json`; use the actual extracted folder name. The command checks each selected checkpoint's SHA256, applies the same 320-pixel letterbox preprocessing, averages the three raw sigmoid scores and applies the frozen cutoff. Its JSON output includes the score, cutoff and binary label. It performs no training or cutoff fitting. The November export is a research artifact with lower frozen source-test accuracy than the original comparator; it has not replaced that comparator as the recommended benchmark reference.

## Audit public prediction tables

The original comparator export preserves its 618 saved, temperature-scaled test scores. The independent CPU jobs export every selected calibration/test prediction and, for the grouped assessment, every neural and dimension-only out-of-fold prediction. Bounded JSON log blocks carry a SHA256 for the exact UTF-8 CSV bytes. `capture_verification.py` refuses missing blocks, changed scores, duplicate images or a missing final report before writing any table.

All three export jobs completed and seven complete prediction tables are captured. Metrics can be checked locally without image files, weights or Kaggle access:

```sh
python experiments/training_20261115/audit_prediction_tables.py --kind primary
python experiments/training_20261115/audit_prediction_tables.py --kind cv
python experiments/training_20261115/audit_results.py
```

These commands require the corresponding completed JSON records and `predictions/*.csv`; they fail rather than invent pending results. The audit uses round-trip float parsing, validates the CSV hashes, recomputes the stored decisions and reported metrics, checks fresh CPU score differences, reproduces the single-model and paired group-bootstrap intervals, and reconstructs paired correctness counts. It does not refit a threshold or promote a model. A consistent record of a failed strict CPU check remains a failed CPU result.

Public table columns are decoded-pixel SHA256, filename proxy group, inherited binary class, model score, frozen binary decision and cutoff, plus fold and fresh CPU fields where applicable. They contain no image pixels or verified patient identities. November scores are raw mean sigmoid scores; the original comparator scores retain its frozen temperature scaling. Model scores are not validated clinical probabilities. Local audit library versions are recorded separately from Kaggle training versions.

## Data review and paper

`similarity_reproduction.json`, `similarity_candidate_pairs.csv` and `similarity_visual_review.json` document a local reproduction of the original perceptual-similarity screen and a limited 24-pair visual review. The 814 candidates, including 269 involving source-test images, are not confirmed duplicates. No new exclusion was made. `external_data_review.json` records potential external resources and unresolved access/provenance requirements; no external predictions were produced.

`test_similarity_review_plan.json`, `test_similarity_review_observations.json` and `test_similarity_visual_review.json` preserve a subsequent review of all 269 test-involving candidates in 45 sheets. No transformed copy was established at the displayed thumbnail resolution. The two passes cover 283 unique candidates; 531 remain unreviewed. `prepare_test_similarity_review.py` checks decoded image hashes while rendering the local panels, and `finalize_test_similarity_review.py` checks recorded review coverage and panel hashes without generating visual decisions. This post hoc, AI-assisted inspection makes no clinical or patient-identity determination and does not change the experiment.

An additional [post hoc cross-fold similarity screen](cv_similarity_screen.json), specified after the accuracy outcomes, compared all 4,850,039 cross-fold pairs within the 3,815 original training images. The unchanged 64-bit pHash rule (Hamming distance at most six) flagged 653 pairs involving 570 images. Independent vectorized and exhaustive scalar calculations agreed. These candidates remain visually unadjudicated and are not confirmed duplicates or shared patients. The earlier reviews across original partitions did not cover this within-training universe. No cohort, model, cutoff or accuracy estimate changed.

The public `cv_similarity_fingerprints.csv` and `cv_similarity_candidate_pairs.csv` allow exhaustive numerical rechecking without image files:

```sh
python experiments/training_20261115/screen_cv_similarity.py --recheck
```

The frozen plan and `cv_similarity_screen_provenance.json` preserve the post hoc timing and source commit. This screen does not adjudicate image identity or estimate accuracy after removing candidates.

The report builder consumes measured records and generates a separate November addendum. The original October report stays unchanged. Plots distinguish development, exploratory source-test and secondary internal-fold evidence. Author declarations, independent clinical review, external validation and journal submission remain outside the completed computational evidence.
