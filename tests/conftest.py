# tests/conftest.py
import os
import sys
import types
import pandas as pd

# 0) Ensure project root is on sys.path for imports
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# 1) Stub out nasdaqdatalink
ndl = types.SimpleNamespace()
ndl.ApiConfig = type("A", (), {"api_key": None})
ndl.get_table = lambda *args, **kwargs: pd.DataFrame()
sys.modules["nasdaqdatalink"] = ndl

# 2) Stub out pandas_market_calendars
# pmc = types.SimpleNamespace()
# class DummyCal:
#     def schedule(self, start_date, end_date):
#         # return empty schedule so completeness_check passes
#         return pd.DataFrame(index=pd.to_datetime([]))
# pmc.get_calendar = lambda name: DummyCal()  # replaved with lambda
import pandas as pd
pmc = types.SimpleNamespace()
class DummyCal:
    def schedule(self, start_date, end_date):
        # generate business-day index between the given dates (inclusive)
        idx = pd.bdate_range(start=start_date, end=end_date)
        return pd.DataFrame(index=idx)
pmc.get_calendar = lambda name: DummyCal()
sys.modules["pandas_market_calendars"] = pmc
