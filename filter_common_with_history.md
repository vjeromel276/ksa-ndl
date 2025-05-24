## Script Name: `filter_common_with_history.py`

**Purpose:**
Filter a raw Sharadar SEP snapshot to a core universe of common‑stock tickers satisfying price, liquidity, and minimum trading‑history requirements.

### Invocation

```bash
python filter_common_with_history.py \
  --date YYYY-MM-DD \
  [--price-threshold FLOAT] \
  [--dollar-vol-threshold FLOAT] \
  [--history-window INT]
```

### Arguments

| Arg                      | Type    | Required | Default   | Description                                                   |
| ------------------------ | ------- | -------- | --------- | ------------------------------------------------------------- |
| `--date`                 | `str`   | Yes      | —         | As‑of date (YYYY-MM-DD) matching the SEP snapshot suffix      |
| `--price-threshold`      | `float` | No       | `5`       | Minimum close price per share                                 |
| `--dollar-vol-threshold` | `float` | No       | `1000000` | Minimum average daily dollar‑volume (volume × close)          |
| `--history-window`       | `int`   | No       | `252`     | Minimum number of unique trading‑day rows required per ticker |

### Constants

| Name                      | Type    | Description                                                                |
| ------------------------- | ------- | -------------------------------------------------------------------------- |
| `TICKER_META_PATH`        | `str`   | Path to ticker metadata Parquet (`sep_dataset/SHARADAR_TICKERS_2.parquet`) |
| `DEFAULT_PRICE_THRESHOLD` | `float` | Default price‐floor for filtering                                          |
| `DEFAULT_DOLLAR_VOL`      | `float` | Default avg dollar‑volume threshold                                        |
| `DEFAULT_WINDOW_DAYS`     | `int`   | Default lookback window in trading days                                    |

### Functions & Contracts

| Function                     | Signature                                                | Returns            | Side Effects                                                                                |
| ---------------------------- | -------------------------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------- |
| `setup_logging()`            | `() -> None`                                             | None               | Configures the root logger at INFO level with timestamped output                            |
| `get_valid_common_tickers()` | `(meta_df: pd.DataFrame) -> pd.Index`                    | Index of tickers   | Filters metadata for common‑stock tickers; logs count                                       |
| `filter_price()`             | `(sep: pd.DataFrame, price_thr: float) -> pd.DataFrame`  | Filtered DataFrame | Applies `sep[sep['close'] >= price_thr]`; logs row counts                                   |
| `filter_dollar_volume()`     | `(sep: pd.DataFrame, dollar_thr: float) -> pd.DataFrame` | Filtered DataFrame | Computes avg daily `close*volume`, retains tickers ≥ threshold; logs ticker and row changes |
| `filter_history_window()`    | `(sep: pd.DataFrame, window: int) -> pd.DataFrame`       | Filtered DataFrame | Keeps only tickers with ≥ `window` unique `date` entries; logs remaining ticker count       |
| `parse_args()`               | `() -> argparse.Namespace`                               | CLI args           | Defines and parses script arguments                                                         |
| `main()`                     | `() -> None`                                             | None               | Orchestrates filtering stages and writes output Parquet                                     |

### Flow Overview

1. **Initialize logging** with `setup_logging()`.
2. **Parse CLI arguments** (`--date`, thresholds, window).
3. **Validate `--date`** format (`YYYY-MM-DD`).
4. **Load** raw SEP Parquet (`sep_dataset/SHARADAR_SEP_<date>.parquet`) and ticker metadata.
5. **Whitelist common stocks** via `get_valid_common_tickers()`.
6. **Apply filters** in sequence:

   1. `filter_price()`
   2. `filter_dollar_volume()`
   3. `filter_history_window()`
7. **Write output** to `sep_dataset/SHARADAR_SEP_filtered_<date>.parquet`.
8. **Log summary** of final row and ticker counts.

### Dependencies

* **Standard library:** `argparse`, `logging`, `os`, `datetime`
* **Third‑party:** `pandas`

### Outputs

* **Filtered SEP Parquet:** `sep_dataset/SHARADAR_SEP_filtered_<date>.parquet` containing only common‑stock tickers that meet price, liquidity, and history requirements.
