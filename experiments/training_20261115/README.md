# November 2026 training and verification

The primary experiment completed nine fresh models. Development accuracy selected the three-seed regularized EfficientNetV2-S ensemble, but its frozen decision rule achieved **83.98% source-test accuracy**, compared with **87.38%** for the original benchmark comparator. No threshold was changed after observing that result. Full CPU reproduction passed on all 1,100 calibration/test images with zero changed classifications. Secondary grouped training is running; execution is tracked in the records below.

This folder contains exploratory research, not clinical validation. All original source cohorts had already been used in earlier experiments. Filename groups are patient proxies, and independent external testing is incomplete.

## Protocol and provenance

| Experiment | Frozen source | Plan | Execution record |
|---|---|---|---|
| Nine primary fits | `submitted_source.py` | `protocol.json` | `run_record.json` |
| Three-fold, three-seed, three-recipe internal assessment | `submitted_grouped_cv.py` | `cv_protocol.json` | `cv_run_record.json` |
| Primary full-cohort CPU reproduction | `submitted_verification.py` | `verification_plan_record.json` | `cpu_run_record.json` |
| Original comparator score export (retrospective, no fitting) | `submitted_comparator_export.py` | `previous_lock.json` | `comparator_export_run_record.json` |

Each submitted file is also embedded in its corresponding `.ipynb`. Source hashes normalize CRLF to LF, matching the verified full Kaggle editor clipboard. Checkpoint and exported-file digests concern exact binary bytes. GitHub commit identifiers in the run records establish the prospective versions.

The original single-seed CV proposal was amended to three seeds before submission and never executed. The first 27-fit commit remained queued and was cancelled after zero execution seconds; `cv_run_record_v1_cancelled.json` preserves that attempt. The retry uses the identical training source. Neither the retry nor the separate CPU checks changes the primary model.

## Reproduce in Kaggle

1. Import `pneumonia_training.ipynb`, enable a GPU and Internet, and attach the original Chest X-Ray Images (Pneumonia) dataset and the original completed 320-pixel benchmark output. The code validates input cohort identities, configuration and checkpoint hashes before fitting.
2. Use Save & Run All. Training fits nine models, selects a development ensemble, freezes its calibration-only cutoff and checkpoints, then scores the previously inspected source test. The 225-minute training cap excludes auditing and final evaluation; early stopping can use much less time.
3. Attach that completed output and the original dataset to `pneumonia_cpu_verification.ipynb`, use CPU, and run it. The packaged verification is pinned to the observed configuration hash. A new training rerun requires rebuilding the verifier for its newly observed frozen hash; the existing execution record must remain historical.
4. The separate `pneumonia_grouped_cv.ipynb` uses only the original training cohort, trains 27 fresh models, and produces out-of-fold scores. Its three recipe ensembles use inner-only checkpoint and cutoff selection. It does not select a replacement primary model or score the original selection, calibration or test images.

Required inputs are discovered beneath `/kaggle/input`; ambiguous or mismatched artifacts cause an error. Complete weight archives remain in the Kaggle outputs, subject to notebook visibility. They are not included in this repository.

## Audit public prediction tables

The original comparator export preserves its 618 saved, temperature-scaled test scores. The independent CPU jobs export every selected calibration/test prediction and, for the grouped assessment, every neural and dimension-only out-of-fold prediction. Bounded JSON log blocks carry a SHA256 for the exact UTF-8 CSV bytes. `capture_verification.py` refuses missing blocks, changed scores, duplicate images or a missing final report before writing any table.

After those jobs complete and their exports are captured, metrics can be checked locally without image files, weights or Kaggle access:

```sh
python experiments/training_20261115/audit_prediction_tables.py --kind primary
python experiments/training_20261115/audit_prediction_tables.py --kind cv
python experiments/training_20261115/audit_results.py
```

These commands require the corresponding completed JSON records and `predictions/*.csv`; they fail rather than invent pending results. The audit uses round-trip float parsing, validates the CSV hashes, recomputes the stored decisions and reported metrics, checks fresh CPU score differences, reproduces the single-model and paired group-bootstrap intervals, and reconstructs paired correctness counts. It does not refit a threshold or promote a model. A consistent record of a failed strict CPU check remains a failed CPU result.

Public table columns are decoded-pixel SHA256, filename proxy group, inherited binary class, model score, frozen binary decision and cutoff, plus fold and fresh CPU fields where applicable. They contain no image pixels or verified patient identities. November scores are raw mean sigmoid scores; the original comparator scores retain its frozen temperature scaling. Model scores are not validated clinical probabilities. Local audit library versions are recorded separately from Kaggle training versions.

## Data review and paper

`similarity_reproduction.json`, `similarity_candidate_pairs.csv` and `similarity_visual_review.json` document a local reproduction of the original perceptual-similarity screen and a limited 24-pair visual review. The 814 candidates, including 269 involving source-test images, are not confirmed duplicates. No new exclusion was made. `external_data_review.json` records potential external resources and unresolved access/provenance requirements; no external predictions were produced.

The report builder consumes measured records and generates a separate November addendum. The original October report stays unchanged. Plots distinguish development, exploratory source-test and secondary internal-fold evidence. Author declarations, independent clinical review, external validation and journal submission remain outside the completed computational evidence.
