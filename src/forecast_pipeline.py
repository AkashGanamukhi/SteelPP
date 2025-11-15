from __future__ import annotations

import os
from datetime import date

import pandas as pd

from config import (
    OUTPUT_DIR,
    FORECAST_HORIZON_DAYS,
)
from src.data_loader import load_weekly_price_data
from src.models.time_series_model import TimeSeriesANN
from src.models.sentiment_model import NewsSentimentModel
from src.models.fusion_model import (
    compute_sentiment_scalar,
    apply_sentiment_to_forecast,
    compute_10day_bracket_ranges,
)
from src.services.perplexity_client import fetch_steel_news
from src.utils.plotting import plot_forecast


def run_forecast(
    data_path: str,
    current_date: date,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    use_news: bool = True,
):
    # 1. Load weekly HRC India price data
    df_weekly = load_weekly_price_data(data_path)

    # 2. Time-series base model (ANN)
    ts_model = TimeSeriesANN()
    daily_forecast = ts_model.forecast_daily(
        df_weekly=df_weekly,
        current_date=current_date,
        horizon_days=horizon_days,
    )  # columns: date, base_price

    # 3. Sentiment from news (optional)
    sentiment_scalar = 0.0
    sentiment_df = None
    if use_news:
        news_items = fetch_steel_news()
        sentiment_model = NewsSentimentModel()
        sentiment_df = sentiment_model.score_news_items(news_items)
        sentiment_scalar = compute_sentiment_scalar(sentiment_df)

    # 4. Apply fusion
    final_forecast = apply_sentiment_to_forecast(
        daily_forecast,
        sentiment_scalar=sentiment_scalar,
    )

    # 5. Bracket ranges
    bracket_ranges = compute_10day_bracket_ranges(final_forecast)

    # 6. Save CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_filename = f"forecast_{current_date.strftime('%Y%m%d')}.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    try:
        final_forecast.to_csv(csv_path, index=False)
    except PermissionError:
        # If the default filename is locked by another process, write a timestamped file instead
        import time

        alt_csv_path = os.path.join(
            OUTPUT_DIR, f"forecast_{current_date.strftime('%Y%m%d')}_{int(time.time())}.csv"
        )
        final_forecast.to_csv(alt_csv_path, index=False)
        print(f"Warning: could not write to {csv_path}; saved to {alt_csv_path}")
        csv_path = alt_csv_path

    # 7. Plot
    plot_path = plot_forecast(final_forecast, current_date=current_date)

    return final_forecast, bracket_ranges, plot_path, csv_path
