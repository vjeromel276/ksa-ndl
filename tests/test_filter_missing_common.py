import json
import pytest
from filter_missing_common import main as filter_main

def test_filter_missing_common(tmp_path, monkeypatch):
    data = {
        "A": ["2025-01-01","2025-01-02"],
        "B.U": ["2025-01-01"],
        "CWS": ["2025-01-02"]
    }
    inp = tmp_path / "missing_pairs.json"
    out = tmp_path / "missing_pairs_common.json"
    with open(inp, "w") as f:
        json.dump(data, f)
    monkeypatch.chdir(tmp_path)

    filter_main([])

    with open(out) as f:
        filtered = json.load(f)
    # Only "A" should survive
    assert list(filtered.keys()) == ["A"]
    assert filtered["A"] == ["2025-01-01","2025-01-02"]
