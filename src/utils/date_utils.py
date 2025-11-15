from datetime import date, timedelta
import pandas as pd


def date_range_daily(start_date: date, end_date: date) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, end=end_date, freq="D")


def add_days(d: date, n: int) -> date:
    return d + timedelta(days=n)
