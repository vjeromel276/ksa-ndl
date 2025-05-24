## Script Name: `daily_pipeline.py`

**Purpose:**
Orchestrate the end‑to‑end daily data pipeline: download and merge fresh SEP data, promote to gold master, filter to common stocks with price/liquidity/history requirements, compute coverage & volume metrics, and produce a clean ticker universe snapshot.

### Invocation

```bash
python daily_pipeline.py --date YYYY-MM-DD
```

### Arguments

| Arg      | Type  | Required | Default | Description                             |
| -------- | ----- | -------- | ------- | --------------------------------------- |
| `--date` | `str` | Yes      | —       | Trading date in `YYYY-MM-DD` to process |

### Functions & Contracts

| Function                       | Signature                   | Returns  | Side Effects                                                                                                                   |
| ------------------------------ | --------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `parse_args()`                 | `() -> argparse.Namespace`  | CLI args | Defines and parses `--date` argument                                                                                           |
| `download_and_merge(date)`     | `(str) -> Path`             | Path     | Calls `daily_download.py --date date --merge`; returns path to `SHARADAR_SEP_<date>.parquet`                                   |
| `promote_to_gold_master(path)` | `(Path) -> None`            | None     | Reads master `_2.parquet`, appends new daily CSV, dedupes, writes date‑stamped and overwrites gold                             |
| `filter_common(date)`          | `(str) -> Path`             | Path     | Invokes `filter_common_tickers.py --date date`; returns path to `SHARADAR_SEP_common_<date>.parquet`                           |
| `compute_coverage(date)`       | `(str) -> (Path,Path,Path)` | Paths    | Runs `compute_coverage_and_volume.py --common-sep ...`; writes coverage, cov+vol, and clean universe CSVs; returns their paths |
| `filter_with_history(date)`    | `(str) -> Path`             | Path     | Executes `filter_common_with_history.py --date date`; returns `SHARADAR_SEP_filtered_<date>.parquet`                           |
| `main()`                       | `() -> None`                | None     | Orchestrates all stages in sequence; logs start/end and individual step success                                                |

### Flow Overview

1. **Initialize logging** (INFO level).
2. **Parse `--date`** argument.
3. **Download & Merge**: Run download script and merge into Parquet snapshot.
4. **Promote to Gold Master**: Append daily data to master `_2.parquet`.
5. **Filter to common stock**: Apply price & liquidity filters via `filter_common_tickers.py`.
6. **Compute Coverage & Volume**: Generate coverage metrics and clean universe CSVs with `compute_coverage_and_volume.py`.
7. **Filter with History**: Apply 252‑day history window via `filter_common_with_history.py`.
8. **Completion log**: Confirm all artifacts and successful pipeline run.

### Dependencies

* **CLI scripts**: `daily_download.py`, `merge_daily_download.py`, `filter_common_tickers.py`, `compute_coverage_and_volume.py`, `filter_common_with_history.py`
* **Standard library**: `argparse`, `logging`, `subprocess` or `importlib` for script invocations
* **Environment**: Paths must align (`data/sharadar_daily`, `sep_dataset`, `models`, `predictions`).

### Outputs

* **Daily CSV**: `data/sharadar_daily/SHARADAR_SEP_<date>.csv`
* **Parquet snapshots**:

  * `sep_dataset/SHARADAR_SEP_<date>.parquet`
  * `sep_dataset/SHARADAR_SEP_2.parquet` (gold master)
  * `sep_dataset/SHARADAR_SEP_common_<date>.parquet`
  * `sep_dataset/SHARADAR_SEP_filtered_<date>.parquet`
* **Coverage & Universe CSVs**:

  * `sep_dataset/ticker_coverage_<date>.csv`
  * `sep_dataset/ticker_coverage_vol_<date>.csv`
  * `sep_dataset/ticker_universe_clean_<date>.csv`

This spec aligns the orchestrator with each underlying script’s contract, providing a clear, reproducible daily pipeline blueprint.
