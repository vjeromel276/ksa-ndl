import json
import pandas as pd
import pytest
from export_missing import main as export_main

def test_export_missing_creates_json(tmp_path, monkeypatch):
    # 1) Create a tiny SEP with a known hole
    sep = pd.DataFrame({
        "ticker": ["X", "X", "Y"],
        "date":   pd.to_datetime(["2025-01-01","2025-01-02","2025-01-01"])
    })
    in_path = tmp_path / "SEP.parquet"
    sep.to_parquet(in_path, index=False)
    monkeypatch.setenv("SEP_MASTER", str(in_path))

    # 2) Run export_missing (assumes it reads SEP_MASTER & writes missing_pairs.json)
    out_json = tmp_path / "missing_pairs.json"
    monkeypatch.chdir(tmp_path)  # so it writes in tmp_path
    export_main([])

    # 3) Validate JSON contents
    with open(out_json) as f:
        mp = json.load(f)
    # Expect Y missing on 2025-01-02
    assert mp == {"Y": ["2025-01-02"]}
