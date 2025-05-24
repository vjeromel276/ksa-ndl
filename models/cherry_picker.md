## Script Name: `models/cherry_picker.py`

**Purpose:**
Utility module for selecting tickers whose trading histories meet the minimum lookback requirements for a given prediction horizon.

### Invocation

Import and call programmatically (no standalone CLI):

```python
from models.cherry_picker import get_valid_tickers_for_horizon

valid_tickers = get_valid_tickers_for_horizon(
    universe_csv="ticker_universe_clean_2025-05-22.csv",
    horizon="5d"
)
```

(Commonly invoked within `train_and_save_models.py` to cherry-pick the universe.)

### Arguments

| Arg            | Type  | Required | Description                                                                 |
| -------------- | ----- | -------- | --------------------------------------------------------------------------- |
| `universe_csv` | `str` | Yes      | Path to `ticker_universe_clean_<date>.csv` containing `have_days`.          |
| `horizon`      | `str` | Yes      | One of `"1d"`, `"5d"`, `"10d"`, or `"30d"` representing prediction horizon. |

### Constants & Contracts

| Name            | Type            | Description                                                           |
| --------------- | --------------- | --------------------------------------------------------------------- |
| `_HORIZON_DAYS` | `Dict[str,int]` | Maps each horizon to required minimum trading days (e.g. `"5d":260`). |

### Functions & Contracts

| Function                        | Signature                                        | Returns     | Side Effects                                                                                        |
| ------------------------------- | ------------------------------------------------ | ----------- | --------------------------------------------------------------------------------------------------- |
| `get_valid_tickers_for_horizon` | `(universe_csv: str, horizon: str) -> List[str]` | `List[str]` | Reads the CSV with `pd.read_csv`, parses date columns; raises `ValueError` if `horizon` is invalid. |

### Flow Overview

1. **Validate Horizon:** Check `horizon` exists in `_HORIZON_DAYS`; raise on invalid key.
2. **Read Universe CSV:** Load into DataFrame with `parse_dates` for `listed`, `delisted`, `win_start`, and `win_end`.
3. **Filter by History:** Compute `min_days = _HORIZON_DAYS[horizon]` and select tickers where `have_days >= min_days`.
4. **Return Unique List:** Extract `.unique().tolist()` of the filtered tickers.

### Dependencies

* **Python standard library:** `typing`
* **Third-party:** `pandas`
* **Integration:** Used by `train_and_save_models.py` for horizon-based ticker selection (currently optional).

### Outputs

* **Return value:** `List[str]` of tickers meeting minimum trading-day requirement for the specified horizon.
