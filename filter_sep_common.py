#!/usr/bin/env python3
import pandas as pd

def filter_sep_common(df: pd.DataFrame, universe_csv: str) -> pd.DataFrame:
    """
    Given a full SEP DataFrame and a CSV of approved tickers,
    return only the rows for those tickers.
    """
    universe = pd.read_csv(universe_csv)["ticker"].astype(str).unique()
    return df[df["ticker"].isin(universe)].reset_index(drop=True)

# CLI entrypoint
def main():
    import argparse, os
    p = argparse.ArgumentParser(
        description="Filter a full SEP Parquet to only your clean common‐stock universe"
    )
    p.add_argument("sep_parquet", help="Input SHARADAR_SEP.parquet")
    p.add_argument("universe",    help="CSV of approved tickers")
    p.add_argument(
        "--out", help="Output SHARADAR_SEP_common.parquet", default="sep_dataset/SHARADAR_SEP_common.parquet"
    )
    args = p.parse_args()

    df = pd.read_parquet(args.sep_parquet)
    common_df = filter_sep_common(df, args.universe)
    common_df.to_parquet(args.out, index=False)
    print(f"Wrote {args.out} ({len(common_df)} rows × {len(common_df.columns)} cols)")

if __name__ == "__main__":
    main()
