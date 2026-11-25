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

Use an isolated Linux or WSL2 Python environment with CUDA 13, `cuml-cu13==26.10.*`, and Matplotlib. See the [RAPIDS installation guide](https://docs.nvidia.com/datascience/install/) for runtime setup. Run these commands from the repo root in that environment:

```bash
python src/train_models.py --cuda --predict-test
python src/evaluate_models.py data/processed/training_cuda_with_test
```

With `--cuda`, the three learned estimators fit and predict through native cuML using GPU feature arrays. CSV loading, feature preprocessing, the dummy baseline, and scoring use the CPU. Omit `--predict-test` for a validation-only run. Omit `--cuda` to use scikit-learn estimators on CPU and use `training_with_test` for the evaluation path.

The run folder contains fitted models, split fingerprints, settings and validation scores in `run.json`, and predictions. `row_index` is the zero-based row position in the split. Model pickle files return numeric class IDs in the order listed in `run.json`. Load pickle files only from trusted sources.

The `evaluation/` folder contains test and per-class metrics plus annotated confusion matrices in CSV and PNG. The benign false-alarm rate measures the fraction of benign rows predicted as attacks. Macro and weighted false-alarm rates aggregate the one-vs-rest false-positive rates across all classes. Latency includes feature preprocessing and host/device transfers for in-memory inputs. Batch latency is the median of five full-test predictions divided by the number of rows. Single-record latency uses 100 sampled rows after warm-up.

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
