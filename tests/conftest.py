# tests/conftest.py
import sys
import types
import pandas as pd

# 1) Stub out nasdaqdatalink
ndl = types.SimpleNamespace()
ndl.ApiConfig = type("A", (), {"api_key": None})
ndl.get_table = lambda *args, **kwargs: pd.DataFrame()
sys.modules["nasdaqdatalink"] = ndl

# 2) Stub out pandas_market_calendars
pmc = types.SimpleNamespace()
class DummyCal:
    def schedule(self, start_date, end_date):
        # return empty schedule so ingest_sharadar_day's trading-day check passes
        return pd.DataFrame(index=pd.to_datetime([]))
pmc.get_calendar = lambda name: DummyCal()
sys.modules["pandas_market_calendars"] = pmc
