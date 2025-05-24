## Script Name: `merge_daily_download.py`

**Purpose:** Merge a daily SHARADAR SEP CSV into a date-stamped Parquet snapshot and optionally overwrite the gold master file.

### Invocation

```bash
python merge_daily_download.py --date YYYY-MM-DD [--master-dir PATH] [--data-dir PATH] [--output-dir PATH] [--update-gold]
```

### Arguments

| Arg             | Type   | Required | Default               | Description                                                                              |
| --------------- | ------ | -------- | --------------------- | ---------------------------------------------------------------------------------------- |
| `--date`        | `str`  | Yes      | —                     | Trading date in `YYYY-MM-DD` format for the daily CSV                                    |
| `--master-dir`  | `str`  | No       | `sep_dataset`         | Directory containing the master `SHARADAR_<table>_2.parquet`                             |
| `--data-dir`    | `str`  | No       | `data/sharadar_daily` | Directory where the daily CSV (`SHARADAR_<table>_<date>.csv`) is stored                  |
| `--output-dir`  | `str`  | No       | `sep_dataset`         | Directory to write the date-stamped Parquet snapshot (`SHARADAR_<table>_<date>.parquet`) |
| `--update-gold` | `bool` | No       | `False`               | If set, also overwrite the gold master Parquet with the merged snapshot                  |

### Functions & Contracts

| Function          | Signature                                                                                             | Returns | Side Effects                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `setup_logging()` | `() -> None`                                                                                          | None    | Configures root logger to DEBUG level with timestamped messages                                                                                                  |
| `merge_table()`   | `(table: str, master_dir: str, data_dir: str, output_dir: str, date: str, update_gold: bool) -> None` | None    | Loads master Parquet; reads daily CSV; concatenates; deduplicates on `[ticker,date]`; sorts; writes snapshot; optionally overwrites master fileciteturn4file0 |

### Flow Overview

1. **Initialize logging** via `setup_logging()`.
2. **Parse CLI arguments** (`date`, `--master-dir`, `--data-dir`, `--output-dir`, `--update-gold`).
3. **Ensure output directory exists** with `os.makedirs(args.output_dir, exist_ok=True)`.
4. **Invoke `merge_table()`** with:

   * `table="SEP"`
   * `master_dir=args.master_dir`
   * `data_dir=args.data_dir`
   * `output_dir=args.output_dir`
   * `date=args.date`
   * `update_gold=args.update_gold`
5. **`merge_table()`**:

   * Load the master Parquet from `master_dir/SHARADAR_<table>_2.parquet`.
   * Read daily CSV from `data_dir/SHARADAR_<table>_<date>.csv`.
   * Concatenate DataFrames and drop duplicates on `[ticker, date]`.
   * Coerce `date` column to datetime and sort by `[ticker, date]`.
   * Write a new date-stamped Parquet snapshot to `output_dir`.
   * If `update_gold=True`, overwrite the master Parquet with the combined DataFrame.

### Dependencies

* **Standard library**: `argparse`, `logging`, `os`
* **Third-party**: `pandas`

### Outputs

* **Date-stamped Parquet**: `sep_dataset/SHARADAR_SEP_<date>.parquet`
* **(Optional) Gold master overwrite**: `sep_dataset/SHARADAR_SEP_2.parquet` if `--update-gold` is set
