import pandas as pd
import pytest
import json
from make_complete import main as make_main

def test_make_complete_appends(tmp_path, monkeypatch):
    # 1) Create master SEP with one date
    master = pd.DataFrame({
        "ticker": ["Z"], "date": pd.to_datetime(["2025-01-01"]), "open":[1]
    })
    master_path = tmp_path / "SEP_master.parquet"
    master.to_parquet(master_path, index=False)

    # 2) missing_pairs.json with a single (Z,2025-01-02)
    mp = {"Z": ["2025-01-02"]}
    with open(tmp_path/"missing_pairs.json","w") as f:
        json.dump(mp, f)

    # 3) Stub out API to return a one-row DF for Z@2025-01-02
    def fake_get_table(table, ticker, date, paginate):
        return pd.DataFrame({"ticker":["Z"],"date":[date],"open":[2]})
    monkeypatch.setenv("NASDAQ_API_KEY","DUMMY")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MASTER_PATH", str(master_path))
    import nasdaqdatalink
    monkeypatch.setattr(nasdaqdatalink, "get_table", fake_get_table)

    # 4) Run make_complete
    make_main([])

    # 5) Validate the master now has two rows
    df2 = pd.read_parquet(master_path)
    assert set(df2["date"].astype(str)) == {"2025-01-01","2025-01-02"}
