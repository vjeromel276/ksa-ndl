#!/usr/bin/env python3
import os
import json
import argparse
import re

"""
filter_missing_common.py

Read missing_pairs.json, drop non-common-stock tickers (units/warrants/etc),
and write missing_pairs_common.json.

Usage:
  python filter_missing_common.py \
    [--in path/to/missing_pairs.json] \
    [--out path/to/missing_pairs_common.json]
"""

def main(args=None):
    p = argparse.ArgumentParser(description="Filter missing pairs to common-stock tickers")
    p.add_argument(
        "--in",
        dest="input",
        help="Input missing_pairs.json (overrides MISSING_JSON env var)",
        default=None
    )
    p.add_argument(
        "--out",
        dest="output",
        help="Output common-stock JSON path (defaults to missing_pairs_common.json)",
        default="missing_pairs_common.json"
    )
    opts = p.parse_args(args or [])

    inp = opts.input or os.environ.get("MISSING_JSON", "missing_pairs.json")
    outp = opts.output

    # Load the full missing map
    with open(inp) as fp:
        missing_map = json.load(fp)

    # Keep only tickers matching common-stock patterns:
    #  - no dot suffix (e.g. .U, .WS)
    #  - no trailing letter codes like W, R
    # Keep only plain common‐stock tickers (no dot suffix, no unit/warrant/right suffix)
    common = {}
    drop_sufs = ("U", "R", "W", "WS")
    for ticker, dates in missing_map.items():
        # must be alphanumeric
        if not re.fullmatch(r"[A-Z0-9]+", ticker):
            continue
        # drop any ending in those suffixes
        if any(ticker.endswith(suf) for suf in drop_sufs):
            continue
        common[ticker] = dates

    # Write out filtered JSON
    with open(outp, "w") as fp:
        json.dump(common, fp, indent=2)

    print(f"Filtered {len(common)}/{len(missing_map)} tickers to {outp}")

if __name__ == "__main__":
    main()
