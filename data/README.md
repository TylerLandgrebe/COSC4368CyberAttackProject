# Dataset

The proposal names Kaggle's [Cybersecurity Threat Detection Dataset](https://www.kaggle.com/datasets/dhrubangtalukdar/cybersecurity-threat-detection-dataset).

Download the CSV from Kaggle and place it in this directory as `data/raw/cybersecurity_dataset.csv` (or pass its actual location with `--input`). The repository intentionally does not include the downloaded data. Do not commit raw or processed dataset files; `.gitignore` excludes `data/raw/` and generated files in `data/processed/`.

## Preparation behavior

- Source data is loaded without changing its values; blank strings are treated as missing values.
- Exact duplicate rows and missing values are counted in the summary, not silently removed or filled.
- Rows missing the selected target are excluded because they cannot be used for supervised training.
- The default target is `attack_type`; choose `label` for the binary task.
- Both `label` and `attack_type` are withheld from the feature tables to prevent target leakage.
- The target is stratified into reproducible 70% training, 15% validation, and 15% testing splits (seed 42 by default).
- Imputation, scaling, and categorical encoding are fit on the training split only. The test split should be kept untouched until model selection is complete.
- Missing categorical values are encoded as their own category by the supplied preprocessing helper rather than replaced with the most common value.

The proposal describes a Kaggle dataset with network-flow features and safe/malicious and attack-type labels. The prep script validates the selected target but does not hard-code an entire feature schema; it records all available columns in the summary so that changes in the source file are visible.
