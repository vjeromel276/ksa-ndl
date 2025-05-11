#!/usr/bin/env python3
import json

# 1) Load your full missing‐pairs map
with open("missing_pairs.json") as fp:
    missing_map = json.load(fp)

# 2) Classification function (same as before)
def is_common(ticker: str) -> bool:
    t = ticker.upper()
    # strip common suffixes for non‐common shares
    if t.endswith(('.U','U','.WS','W','.R','R','.P','P')):
        return False
    return True

# 3) Filter down to only “common” tickers
common_map = {
    ticker: dates
    for ticker, dates in missing_map.items()
    if is_common(ticker)
}

# 4) Write out the new JSON
with open("missing_pairs_common.json", "w") as fp:
    json.dump(common_map, fp, indent=2)

print(f"Kept {len(common_map)} tickers for common‐stock backfill.")
print("Output written to missing_pairs_common.json")
