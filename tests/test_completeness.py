import pandas as pd
import pytest
from completeness_check import main as cc_main


def test_no_missing(tmp_path, monkeypatch, capsys):
    # create a SEP master with full calendar coverage
    sep = pd.DataFrame({
        "ticker":["AAA","AAA","BBB","BBB"],
        "date": pd.to_datetime(["2025-01-01","2025-01-02","2025-01-01","2025-01-02"])
    })
    sep.to_parquet(tmp_path/"sep.parquet", index=False)
    monkeypatch.setenv("SEP_MASTER", str(tmp_path/"sep.parquet"))

    # run completeness check
    cc_main()
    captured = capsys.readouterr()
    assert "Missing pairs: 0" in captured.out


def test_some_missing(tmp_path, monkeypatch, capsys):
    # only AAA on 01 and BBB on 02
    sep = pd.DataFrame({
        "ticker":["AAA","BBB"],
        "date": pd.to_datetime(["2025-01-01","2025-01-02"])
    })
    sep.to_parquet(tmp_path/"sep.parquet", index=False)
    monkeypatch.setenv("SEP_MASTER", str(tmp_path/"sep.parquet"))

    cc_main()
    out = capsys.readouterr().out
    # expect two missing pairs: AAA@02 and BBB@01
    assert "Missing pairs: 2" in out
