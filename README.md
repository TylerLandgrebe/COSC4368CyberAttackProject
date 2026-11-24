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

The default target is `attack_type`, matching the proposal's multi-class evaluation. Use `--target label` for the binary benign/malicious task. The script writes stratified 70/15/15 CSV splits and a data summary under `data/processed/`. Both `label` and `attack_type` are excluded from model features to prevent one target from revealing the other.

Feature encoding and imputation should be fitted inside a scikit-learn pipeline using only the training split. `src/preprocessing.py` provides a reusable training-only preprocessor; missing categorical values are kept as an explicit category. It does not drop high-cardinality columns automatically; inspect `data_summary.json` and document any feature exclusions before training.

See [data/README.md](data/README.md) for data handling and provenance notes.
