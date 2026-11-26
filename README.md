# COSC 4368 Cybersecurity Threat Detection

Project code for preparing a labeled network-traffic dataset and evaluating supervised classifiers for cybersecurity threat detection.

## Dataset preparation

1. Download the CSV from the Kaggle dataset linked in [data/README.md](data/README.md).
2. Put the CSV in `data/raw/` (the raw data directory is ignored by Git).
3. Install the packages in `requirements.txt`.
4. Run the preparation script, passing the CSV's actual filename:

   ```powershell
   python -m pip install -r requirements.txt
   python src/prepare_data.py --input data/raw/cybersecurity_dataset.csv
   ```

The default target is `attack_type`, matching the proposal's multiclass evaluation. The script writes stratified 70/15/15 CSV splits and a data summary under `data/processed/`. Both `label` and `attack_type` are excluded from model features. For the binary experiment below, keep these splits so the comparison uses the same records.

Feature encoding and imputation should be fitted inside a scikit-learn pipeline using only the training split. `src/preprocessing.py` provides a reusable training-only preprocessor; missing categorical values are kept as an explicit category. It does not drop high-cardinality columns automatically; inspect `data_summary.json` and document any feature exclusions before training.

See [data/README.md](data/README.md) for data handling and provenance notes.

## Multiclass training

From the repository root, run `python src/train_models.py` after preparing the default `attack_type` splits. The experiment fits a most-frequent baseline and compares class-balanced logistic regression, an RBF SVM, and a random forest with and without `timestamp`, `src_ip`, and `dst_ip`. Encoders, imputers, scalers, and classifiers are fitted on the training split. The SVM uses inverse-frequency class weights, and the other learned models receive inverse-frequency sample weights.

### CUDA training and test evaluation

Use an isolated Linux or WSL2 Python environment with CUDA 13, `cuml-cu13==26.10.*`, Matplotlib, and `xgboost==3.4.1`. See the [RAPIDS installation guide](https://docs.nvidia.com/datascience/install/) for runtime setup. Run these commands from the repo root in that environment:

```bash
python src/train_models.py --cuda --predict-test --include-xgboost
python src/eval.py
```

With `--cuda`, logistic regression, SVM, and random forest fit and predict through native cuML using GPU feature arrays. The optional direct XGBoost classifier also runs on CUDA. CSV loading, feature preprocessing, the dummy baseline, and scoring use the CPU. Omit `--predict-test` for a validation-only run. The evaluator verifies fingerprints for both test features and labels before scoring.

The run folder contains fitted models, split fingerprints, settings and validation scores in `run.json`, and predictions. `row_index` is the zero-based row position in the split. Model pickle files return numeric class IDs in the order listed in `run.json`. Load pickle files only from trusted sources.

The `evaluation/` folder contains test and per-class metrics plus annotated confusion matrices in CSV and PNG. The benign false-alarm rate measures the fraction of benign rows predicted as attacks. Macro and weighted false-alarm rates aggregate the one-vs-rest false-positive rates across all classes. Latency includes feature preprocessing and host/device transfers for in-memory inputs. Batch latency is the median of five full-test predictions divided by the number of rows. Single-record latency uses 100 sampled rows after warm-up.

### Direct ten-class XGBoost

Depth 6 won the validation macro-F1 sweep over depths 4, 6, and 8. XGBoost and the forest below use the same data split and omit timestamps and IPs. The original holdout was reused for this follow-up.

| Classifier | Test macro-F1 | Balanced accuracy | Correct attack type | Benign false alarms | Median single-record latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random forest | 0.245 | 30.8% | 17/60 | 77/1,440 | 7.06 ms |
| Direct XGBoost | 0.306 | 40.4% | 30/60 | 122/1,440 | 11.94 ms |

Direct XGBoost is the strongest multiclass model in this run by macro-F1. Its default predictions produce more false alarms than the forest. For binary triage, the separate XGBoost gate below evaluates thresholds chosen to control false alarms.

## Binary attack detection

```bash
python src/train_detector.py --cuda --predict-test
python src/train_detector.py --cuda --chronological
```

The detector derives the binary target from the existing `attack_type` rows. On this dataset, every non-benign row has `label=1` and every benign row has `label=0`. The candidates are a depth-16 forest using the existing features without timestamp and IPs, plus logistic regression and two shallower forests using categorical destination ports, URL summary features, and the log byte ratio. The engineered version drops source port and raw URLs.

For each false-alarm budget (1% and 5%), validation chooses the candidate and threshold that catch the most attacks within budget. Ties prefer fewer false alarms and then higher average precision. The chosen thresholds are applied as-is to evaluation rows. Average precision measures score quality across thresholds and is reported alongside the counts at each selected operating point.

The first command writes `data/processed/detector_cuda_original/`. Its test results are exploratory because the original holdout was already used for the multiclass evaluation. The second command writes `data/processed/detector_cuda_chronological/`. It re-splits only the original development rows in timestamp order, using the earliest 70% for fitting, the next 15% for validation, and the latest 15% for evaluation. This checks temporal transfer within the same dataset. In the chronological output, `row_index` identifies the row in the concatenation of the original train and validation splits. Each folder records the candidate metrics, selected model and thresholds, saved model, and evaluation predictions.

### RTX 3060 results

| Evaluation | Detector | Attacks flagged | Benign false alarms | Average precision |
| --- | --- | ---: | ---: | ---: |
| Original holdout | Multiclass forest, any attack prediction | 32/60 (53.3%) | 77/1,440 (5.35%) | — |
| Original holdout | Binary forest, 1% validation budget | 18/60 (30%) | 13/1,440 (0.90%) | 0.477 |
| Original holdout | Binary forest, 5% validation budget | 39/60 (65%) | 69/1,440 (4.79%) | 0.477 |
| Later development rows | Binary forest, 1% validation budget | 9/50 (18%) | 10/1,225 (0.82%) | 0.334 |
| Later development rows | Binary forest, 5% validation budget | 27/50 (54%) | 61/1,225 (4.98%) | 0.334 |

Validation selected the forest with the existing features and without timestamp or IPs for both budgets on both splits. The engineered candidates caught fewer attacks at those budgets. The later-period results come from the same source dataset. Performance on independently collected traffic remains unmeasured.

## Two-stage attack attribution

The cascade keeps the validation-selected binary detector as a gate. A specialist trained on attack rows assigns an attack type only when the gate flags a record. The validation sweep compares logistic regression at three regularization settings and random forests at three depths. The criterion is macro-F1 over the nine attack types at the 5% false-alarm budget. Missed attacks and benign records assigned an attack type count against the relevant classes. Validation selected depth 8 on the original split and depth 4 on the chronological split. Each chosen specialist is used at both gate thresholds.

| Evaluation | Classifier | Attacks flagged | Correct attack type | Benign false alarms |
| --- | --- | ---: | ---: | ---: |
| Original holdout | Multiclass forest | 32/60 | 17/60 | 77/1,440 |
| Original holdout | Cascade, 5% validation budget | 39/60 | 20/60 | 69/1,440 |
| Later development rows | Multiclass forest | 26/50 | 18/50 | 86/1,225 |
| Later development rows | Cascade, 5% validation budget | 27/50 | 12/50 | 61/1,225 |

Each split has its own forest baseline fitted on the same training rows as the cascade. The cascade flagged one more attack and raised 25 fewer false alarms in the later-period check. The original forest assigned six more attacks the correct type in that check. We keep the multiclass forest as the proposal's primary attribution model and report the cascade as an exploratory tradeoff.

## Results for the report and slides

- [Multiclass test table](results/final/multiclass_test_metrics.csv) and [per-class metrics](results/final/multiclass_per_class_metrics.csv)
- [Macro-F1 comparison](results/final/multiclass_macro_f1.png), [forest confusion matrix](results/final/random_forest_confusion.png), and [single-record latency](results/final/single_record_latency.png)
- [Dexter's per-class recall heatmap on the CUDA test predictions](results/final/multiclass_recall_heatmap.png), [XGBoost confusion matrix](results/final/xgboost_confusion.png), and [XGBoost depth sweep](results/final/multiclass_xgboost_validation.csv)
- [Binary operating points](results/final/binary_operating_points.csv) and [recall versus false alarms](results/final/binary_false_alarm_tradeoff.png)
- [Cascade validation sweep](results/final/cascade_validation_metrics.csv), [cascade test comparison](results/final/cascade_test_metrics.csv), and [end-to-end attribution](results/final/cascade_attribution.png)

The files in `results/final/` use the CUDA evaluation outputs. The legacy files directly under `results/` contain the earlier validation-only baseline. To reproduce the cascade files, run both binary detector commands and the multiclass CUDA training and evaluation commands above, then run:

```bash
python src/train_cascade.py --cuda
python src/publish_results.py
```

## Random forest versus XGBoost gates

This extension compares two tree-based detectors before the same previously selected attack-type specialist. Both use the same splits, inverse-frequency sample weights, training-only preprocessing, and a three-depth validation sweep. For each model family and feature set, we select depth and threshold on validation to maximize attack recall at the 1% or 5% benign false-alarm budget. These budgets apply to validation. Evaluation false-alarm rates can exceed them.

At the 5% validation budget, using features without timestamps and IPs:

| Evaluation | Binary gate | Attacks flagged | Correct attack type out of all attacks | Benign false alarms |
| --- | --- | ---: | ---: | ---: |
| Original holdout | Random forest | 39/60 | 20/60 | 69/1,440 |
| Original holdout | XGBoost | 50/60 | 26/60 | 46/1,440 |
| Later development rows | Random forest | 25/50 | 11/50 | 67/1,225 |
| Later development rows | XGBoost | 37/50 | 17/50 | 61/1,225 |

The forest in this table was reselected from the matched depth sweep. The earlier fixed-gate cascade table reports the separately selected original gate. After dropping user-agent and URL, XGBoost flagged 33/60 versus the forest's 26/60 on the original holdout, and 20/50 versus 19/50 on the later rows. XGBoost raised more false alarms than the forest with those features removed. The original holdout has been reused and the later rows come from the same CSV. The result describes performance within this dataset.

The original classifiers remain the main multiclass comparison. The [boosting figure](results/final/boosting_comparison.png), [evaluation results](results/final/boosting_test_metrics.csv), and [validation sweep](results/final/boosting_validation_metrics.csv) cover both budgets and both feature sets. Row-level scores and attack predictions for independent checking are stored locally under `data/processed/boosting_comparison_cuda/`.

After training the cascade above, run the extension in the CUDA environment:

```bash
python -m pip install xgboost==3.4.1
python src/compare_boosting.py --cuda
python src/publish_results.py --include-boosting
```
