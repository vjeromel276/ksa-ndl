#!/usr/bin/env python3
import json
import re

def filter_missing_pairs(missing_map: dict) -> dict:
    """
    Keep only plain common‐stock tickers in the missing_map.
    missing_map: {ticker: [date_str, …], …}
    Returns a filtered {ticker: [date_str,…], …}
    """
    drop_sufs = ("U", "R", "W", "WS")
    common = {}
    for ticker, dates in missing_map.items():
        if not re.fullmatch(r"[A-Z0-9]+", ticker):
            continue
        if any(ticker.endswith(suf) for suf in drop_sufs):
            continue
        common[ticker] = dates
    return common

def missing_map_to_df(missing_map: dict):
    """
    Convert a {ticker: [date_str,…]} map into a DataFrame with columns ['ticker','date'].
    """
    import pandas as pd
    rows = []
    for ticker, dates in missing_map.items():
        for d in dates:
            rows.append({"ticker": ticker, "date": pd.to_datetime(d)})
    return pd.DataFrame(rows)

# CLI entrypoint
def main():
    import argparse, os
    p = argparse.ArgumentParser(
        description="Filter missing_pairs.json to only common‐stock tickers"
    )
    p.add_argument("--in",  dest="inp",  default="missing_pairs.json")
    p.add_argument("--out", dest="outp", default="missing_pairs_common.json")
    args = p.parse_args()

    with open(args.inp) as fp:
        missing_map = json.load(fp)
    common_map = filter_missing_pairs(missing_map)
    with open(args.outp, "w") as fp:
        json.dump(common_map, fp, indent=2)
    print(f"Filtered {len(common_map)}/{len(missing_map)} tickers → {args.outp}")

if __name__ == "__main__":
    main()
