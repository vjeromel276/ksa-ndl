## Script Name: `daily_download.py`

**Purpose:** Download the Sharadar SEP CSV for a specified trading date and optionally merge it into the master Parquet snapshot.

### Invocation

```bash
python daily_download.py --date YYYY-MM-DD [--data-dir PATH] [--master-dir PATH] [--output-dir PATH] [--merge]
```

### Arguments

| Arg            | Type   | Required | Default               | Description                                                                              |
| -------------- | ------ | -------- | --------------------- | ---------------------------------------------------------------------------------------- |
| `--date`       | `str`  | Yes      | —                     | Trading date in `YYYY-MM-DD` format (must be an NYSE session)                            |
| `--data-dir`   | `str`  | No       | `data/sharadar_daily` | Directory where the daily CSV will be saved                                              |
| `--master-dir` | `str`  | No       | `sep_dataset`         | Directory containing the master SEP Parquet snapshot                                     |
| `--output-dir` | `str`  | No       | `sep_dataset`         | Directory to write the date-stamped Parquet snapshot                                     |
| `--merge`      | `bool` | No       | `False`               | If set, merge the downloaded CSV into the master Parquet using `merge_daily_download.py` |

### Functions & Contracts

| Function          | Signature                            | Returns | Side Effects                                                                                          |
| ----------------- | ------------------------------------ | ------- | ----------------------------------------------------------------------------------------------------- |
| `setup_logging()` | `() -> None`                         | None    | Configures root logger to INFO level with timestamped messages                                        |
| `download_file()` | `(url: str, dest_path: str) -> None` | None    | Streams HTTP GET response, writes file in chunks; raises on non-200 status codes; logs progress       |
| `main()`          | `() -> None`                         | None    | Orchestrates CLI parsing, trading-day validation, calls `download_file`, and invokes merge if flagged |

### Flow Overview

1. **Initialize logging** via `setup_logging()` to standardize log output.
2. **Parse CLI arguments** with `argparse` and validate `--date`.
3. **Check trading calendar** (using pandas\_market\_calendars) and exit if the date is not a valid NYSE session.
4. **Download SEP CSV** for `--date` from the Sharadar endpoint into `--data-dir/SHARADAR_SEP_<date>.csv`.
5. **Merge step**: if `--merge` is True, import and call `merge_daily_download.merge_table()` to append and dedupe into the Parquet snapshot under `--output-dir`.
6. **Completion log**: emit final INFO logs indicating CSV download and Parquet write success.

### Dependencies

* **Standard library**: `argparse`, `logging`, `os`, `sys`
* **Third‑party**: `requests` (HTTP), `pandas_market_calendars` (NYSE session check)
* **Local modules**: `merge_daily_download` for Parquet merging

### Outputs

* **Daily CSV**: `data/sharadar_daily/SHARADAR_SEP_<date>.csv`
* **Parquet snapshot**: `sep_dataset/SHARADAR_SEP_<date>.parquet`
