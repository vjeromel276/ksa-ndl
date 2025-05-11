import pandas as pd
from pathlib import Path

# List of Sharadar CSV tables to convert
csv_tables = [
    Path("SHARADAR_ACTIONS_2.csv"),
    Path("SHARADAR_INDICATORS_2.csv"),
    Path("SHARADAR_METRICS_2.csv"),
    Path("SHARADAR_TICKERS_2.csv"),
    Path("SHARADAR_SEP_2.csv")
]

converted = []
errors = []

for csv_path in csv_tables:
    try:
        df = pd.read_csv(csv_path)
        parquet_path = csv_path.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        converted.append(parquet_path)
        print(f"✅ Converted {csv_path.name} → {parquet_path.name} (shape: {df.shape})")
    except Exception as e:
        errors.append((csv_path.name, str(e)))
        print(f"❌ Failed to convert {csv_path.name}: {e}")

# Summary
if converted:
    print("\n🎉 Conversion complete for:")
    for p in converted:
        print(f"  • {p.name}")
if errors:
    print("\n⚠️ Errors encountered:")
    for name, err in errors:
        print(f"  • {name}: {err}")
