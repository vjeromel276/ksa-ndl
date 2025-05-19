# export_missing.py
#!/usr/bin/env python3
"""
export_missing.py

Library + CLI to find missing (ticker × date) pairs in your SEP master.

Usage (CLI):
    python export_missing.py \
      [--master path/to/SEP.parquet] \
      [--out path/to/missing_pairs.json]

Library:
    from export_missing import export_missing_map
    missing_map = export_missing_map(df)  # df has ['ticker','date']
"""
import pandas as pd
import pandas_market_calendars as mcal

def export_missing_map(df: pd.DataFrame) -> dict:
    """
    Given a SEP DataFrame with columns ['ticker','date'], returns:
      { ticker: [ISO date strings of missing days], ... }
    """
    # Ensure correct dtypes
    df = df[["ticker", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Build full calendar × ticker index
    nyse = mcal.get_calendar("NYSE")
    start, end = df["date"].min(), df["date"].max()
    sched = nyse.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    trading_days = sched.index.date

    tickers  = df["ticker"].unique()
    full_idx  = pd.MultiIndex.from_product([tickers, trading_days],
                                           names=["ticker", "date"])
    have_idx  = pd.MultiIndex.from_frame(df)
    missing_idx = full_idx.difference(have_idx)

    # Turn into grouped map: ticker -> [iso_date, ...]
    miss = pd.DataFrame(index=missing_idx).reset_index()
    return (miss
            .groupby("ticker")["date"]
            .apply(lambda dates: [d.isoformat() for d in dates])
            .to_dict())

# CLI entrypoint
def main():
    import os, argparse, json
    p = argparse.ArgumentParser(
        description="Export missing ticker×date pairs to JSON"
    )
    p.add_argument("--master",
                   help="Path to SEP master Parquet",
                   default=None)
    p.add_argument("--out",
                   help="Output JSON path",
                   default="missing_pairs.json")
    args = p.parse_args()

    sep_path = args.master or os.environ.get(
        "SEP_MASTER",
        "sep_dataset/SHARADAR_SEP.parquet"
    )
    df = pd.read_parquet(sep_path, columns=["ticker", "date"])
    missing_map = export_missing_map(df)

    with open(args.out, "w") as fp:
        json.dump(missing_map, fp, indent=2)
    print(f"Exported {len(missing_map)} tickers → {args.out}")

if __name__ == "__main__":
    main()
