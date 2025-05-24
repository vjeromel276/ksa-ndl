## Script Name: `filter_common_tickers.py`

**Purpose:**
Filter a raw Sharadar SEP Parquet file to a vetted universe of common‐stock tickers based on ticker metadata, a minimum close price, and a dollar‐volume liquidity threshold.

### Invocation

```bash
python filter_common_tickers.py --date YYYY-MM-DD [--raw-dir PATH] [--meta-path PATH] [--output-dir PATH]
```

### Arguments

| Arg            | Type  | Required | Default                                  | Description                                                        |
| -------------- | ----- | -------- | ---------------------------------------- | ------------------------------------------------------------------ |
| `--date`       | `str` | Yes      | —                                        | Date in `YYYY-MM-DD` format matching the raw SEP Parquet suffix    |
| `--raw-dir`    | `str` | No       | `sep_dataset`                            | Directory containing `SHARADAR_SEP_<date>.parquet`                 |
| `--meta-path`  | `str` | No       | `sep_dataset/SHARADAR_TICKERS_2.parquet` | Path to the ticker metadata Parquet file                           |
| `--output-dir` | `str` | No       | `sep_dataset`                            | Directory to write the filtered output Parquet (`_common_` suffix) |

### Functions & Contracts

| Function                     | Signature                                                      | Returns                | Side Effects                                                                                              |
| ---------------------------- | -------------------------------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------- |
| `setup_logging()`            | `() -> None`                                                   | None                   | Configures root logger to INFO level and consistent timestamped log format                                |
| `get_valid_common_tickers()` | `(meta_df: pd.DataFrame) -> Set[str]`                          | Set of tickers         | Filters metadata for “Common Stock” category, excludes ADRs/ETFs/etc., enforces ticker format; logs count |
| `filter_sep_by_tickers()`    | `(sep_df: pd.DataFrame, ticker_set: Set[str]) -> pd.DataFrame` | Filtered SEP DataFrame | Subsets raw SEP to allowed tickers; logs row/ticker counts before and after                               |
| `filter_by_price()`          | `(sep_df: pd.DataFrame) -> pd.DataFrame`                       | Filtered SEP DataFrame | Applies `close >= PRICE_THRESHOLD`; logs row counts                                                       |
| `filter_by_dollar_volume()`  | `(sep_df: pd.DataFrame) -> pd.DataFrame`                       | Filtered SEP DataFrame | Computes average daily dollar‐volume per ticker, retains those ≥ threshold; logs counts                   |
| `main()`                     | `() -> None`                                                   | None                   | Orchestrates argument parsing, path construction, file checks, filter chain, and output write             |

### Flow Overview

1. **Initialize logging**: Call `setup_logging()` for uniform log formatting.
2. **Parse CLI arguments**: Read `--date`, optional `--raw-dir`, `--meta-path`, `--output-dir`.
3. **Construct paths**:

   * Raw SEP: `<raw-dir>/SHARADAR_SEP_<date>.parquet`
   * Meta: `<meta-path>`
   * Output: `<output-dir>/SHARADAR_SEP_common_<date>.parquet`
4. **Validate inputs**: Ensure raw SEP and metadata files exist; raise `FileNotFoundError` if missing.
5. **Load data**: Read raw SEP and metadata into pandas DataFrames.
6. **Ticker filtering**: Obtain whitelist via `get_valid_common_tickers(meta_df)`.
7. **Apply filters** in sequence:

   1. `filter_sep_by_tickers(sep_df, whitelist)`
   2. `filter_by_price(...)`
   3. `filter_by_dollar_volume(...)`
8. **Write output**: Save the final filtered DataFrame to the output Parquet path.
9. **Completion log**: Emit an INFO summary with the final row and ticker counts.

### Dependencies

* **Standard library**: `argparse`, `logging`, `os`
* **Third‐party**: `pandas`

### Outputs

* **Filtered Parquet**: `sep_dataset/SHARADAR_SEP_common_<date>.parquet` containing only common‐stock tickers with `close ≥ $5` and avg dollar‐volume ≥ \$1 M/day
