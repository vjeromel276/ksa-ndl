## Script Name: `train_and_save_models.py`

**Purpose:**
Train a direction classifier and a return regressor on a filtered SEP dataset, then save the models with date-encoded filenames.

### Invocation

```bash
python train_and_save_models.py --sep-master PATH [--date YYYY-MM-DD] [--horizon {1d,5d,10d,30d}] [--backend {dummy,xgb,torch}] [--device {cpu,gpu}]
```

### Arguments

| Arg            | Type  | Required | Default  | Description                                                                                   |
| -------------- | ----- | -------- | -------- | --------------------------------------------------------------------------------------------- |
| `--sep-master` | `str` | Yes      | —        | Path to the filtered SEP Parquet (e.g., `sep_dataset/SHARADAR_SEP_common_YYYY-MM-DD.parquet`) |
| `--date`       | `str` | No       | Inferred | Date string `YYYY-MM-DD` for naming outputs; inferred from `--sep-master` filename if omitted |
| `--horizon`    | `str` | No       | `5d`     | Prediction horizon (choices: `1d`, `5d`, `10d`, `30d`)                                        |
| `--backend`    | `str` | No       | `xgb`    | Model backend for classification (choices: `dummy`, `xgb`, `torch`)                           |
| `--device`     | `str` | No       | `gpu`    | Compute device for training (choices: `cpu`, `gpu`)                                           |

### Functions & Contracts

| Function                          | Signature                                                                               | Returns      | Side Effects                                                                                                              |
| --------------------------------- | --------------------------------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `parse_args()`                    | `() -> argparse.Namespace`                                                              | CLI args     | Parses and validates input arguments; exits on invalid horizon or backend.                                                |
| `infer_date_from_sep()`           | `(path: str) -> str`                                                                    | `date_str`   | Extracts `YYYY-MM-DD` from the SEP filename; raises `ValueError` if not found.                                            |
| `validate_full_sep()`             | `(sep_df: pd.DataFrame) -> None`                                                        | None         | Checks SEP DataFrame schema; raises if required columns are missing.                                                      |
| `_coerce_sep_dtypes()`            | `(sep_df: pd.DataFrame) -> pd.DataFrame`                                                | Converted DF | Coerces SEP columns to expected dtypes (e.g., datetime, numeric).                                                         |
| `load_features()`                 | `(sep_df: pd.DataFrame) -> pd.DataFrame`                                                | `X`          | Builds cross-sectional feature matrix; may log shape but does not persist.                                                |
| `load_targets()`                  | `(sep_df: pd.DataFrame) -> pd.DataFrame`                                                | `y_df`       | Constructs classification and regression target columns from SEP (with naming `dir_<horizon>` and `return_<horizon>`).    |
| `train_baseline_classification()` | `(X: pd.DataFrame, y: pd.Series, backend: str, device: str, num_classes: int) -> Model` | `clf`        | Trains and returns a classifier object; uses specified backend and device.                                                |
| `train_baseline_regression()`     | `(X: pd.DataFrame, y: pd.Series, backend: str, device: str) -> Model`                   | `reg`        | Trains and returns a regressor object; uses specified backend and device.                                                 |
| `main()`                          | `() -> None`                                                                            | None         | Orchestrates end-to-end: argument parsing, SEP loading/validation, feature/target alignment, training, and model dumping. |

### Flow Overview

1. **Configure logging**: Set INFO level and timestamp format.
2. **Parse CLI args** using `parse_args()`.
3. **Determine date**: Use `--date` or infer via `infer_date_from_sep()`. Validate format.
4. **Set output paths**:

   * Classifier: `models/dir_<horizon>_clf_<date>.joblib`
   * Regressor: `models/return_<horizon>_reg_<date>.joblib`
5. **Load & validate SEP**: Read Parquet, coerce dtypes, run `validate_full_sep()`.
6. **Feature engineering**:

   * Call `load_features(sep)` → raw features `X`.
   * Cast to `float32`, drop NaNs.
7. **Load & align targets**:

   * Call `load_targets(sep)` → `y_df`.
   * Extract classification and regression columns matching the horizon.
   * Reindex to `X.index`, drop mismatches.
8. **Train models**:

   * **Classifier** via `train_baseline_classification()`, then `joblib.dump()`.
   * **Regressor** via `train_baseline_regression()`, then `joblib.dump()`.
9. **Completion log**: Emit final INFO confirming model paths and successful training.

### Dependencies

* **Standard library**: `argparse`, `logging`, `sys`, `os`, `re`, `datetime`
* **Third-party**: `pandas`, `numpy`, `joblib`
* **Local modules**:

  * `models.cherry_picker`: for horizon‐based ticker selection (currently commented out)
  * `core.schema`: `validate_full_sep` and `_coerce_sep_dtypes`
  * `models.data`: `load_features`, `load_targets`
  * `models.baseline`: training functions for classification and regression

### Outputs

* **Classifier artifact**: `models/dir_<horizon>_clf_<date>.joblib`
* **Regressor artifact**: `models/return_<horizon>_reg_<date>.joblib`

**Note:** This script assumes the SEP input is pre-filtered (via `filter_common_tickers.py`) and focuses solely on training and saving models.
