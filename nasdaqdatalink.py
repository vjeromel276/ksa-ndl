# nasdaqdatalink.py

import pandas as pd

class ApiConfig:
    api_key = None

def get_table(*args, **kwargs):
    # return an empty DataFrame by default; your tests monkeypatch this
    return pd.DataFrame()
