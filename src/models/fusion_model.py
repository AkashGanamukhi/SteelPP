from __future__ import annotations

from typing import List, Dict

import numpy as np
import pandas as pd

from config import BRACKET_SIZE_DAYS, SENTIMENT_ALPHA


def compute_sentiment_scalar(sentiment_df: pd.DataFrame) -> float:
    """
    Aggregate sentiment into a single scalar in [-1, 1].
    Currently just mean polarity; you can be fancier if you like.
    """
    if sentiment_df is None or sentiment_df.empty:
        return 0.0
    return float(np.clip(sentiment_df["polarity"].mean(), -1.0, 1.0))


def apply_sentiment_to_forecast(
    daily_forecast: pd.DataFrame,
    sentiment_scalar: float,
    alpha: float = SENTIMENT_ALPHA,
) -> pd.DataFrame:
    """
    Combine base time-series forecast with sentiment.

    final_price(t) = base_price(t) * (1 + alpha * s * t/T)

    - s in [-1, 1] (avg sentiment)
    - t = 1..T (day index)
    - T = horizon length
    """
    df = daily_forecast.copy().reset_index(drop=True)
    T = len(df)
    if T == 0:
        return df

    t_idx = np.arange(1, T + 1)
    sentiment_factor = 1.0 + alpha * sentiment_scalar * (t_idx / T)

    df["sentiment_factor"] = sentiment_factor
    df["final_price"] = df["base_price"] * df["sentiment_factor"]
    return df


def compute_10day_bracket_ranges(
    forecast_df: pd.DataFrame,
    bracket_size: int = BRACKET_SIZE_DAYS,
) -> List[Dict]:
    """
    Split the 30-day forecast into 3 brackets of 10 days
    and compute [min, max] of final_price for each bracket.
    """
    df = forecast_df.sort_values("date").reset_index(drop=True)

    brackets = []
    n = len(df)
    start_idx = 0
    bracket_num = 0
    while start_idx < n:
        end_idx = min(start_idx + bracket_size, n)
        sub = df.iloc[start_idx:end_idx]
        if sub.empty:
            break
        bracket_num += 1
        brackets.append(
            {
                "bracket_index": bracket_num,
                "start_date": sub["date"].iloc[0].date(),
                "end_date": sub["date"].iloc[-1].date(),
                "min_price": float(sub["final_price"].min()),
                "max_price": float(sub["final_price"].max()),
            }
        )
        start_idx = end_idx

    return brackets
